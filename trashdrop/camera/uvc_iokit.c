/*
 * UVC control requests on macOS, through IOKit.
 *
 * Why not libusb: libusb opens a USB device with USBDeviceOpenSeize and sends
 * asynchronous requests. Current macOS refuses that seize -- to root as well --
 * because its own camera driver is attached. That driver claims only the
 * video *interfaces*, not the device, so a plain USBDeviceOpen succeeds for a
 * normal user and synchronous requests on the default control pipe go
 * through. The approach is the one camtint documents working on macOS 26
 * (github.com/bornaware/camtint, MIT licence).
 *
 * Built on first use by trashdrop/camera/iokit.py and called through ctypes.
 * Every function returns 0 or an IOReturn code; Python decides what it means.
 */

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/usb/IOUSBLib.h>
#include <stdint.h>
#include <string.h>

typedef IOUSBDeviceInterface500 **Device;

static Device device = NULL;
static int device_is_open = 0;

static const char *const kClasses[] = {"IOUSBHostDevice", "IOUSBDevice"};

/* Calls visit() for every USB device; stops early when it returns non-zero.
 * Current macOS publishes each device under both class names, so the legacy
 * one is only searched when the modern one yields nothing -- otherwise every
 * camera would be listed twice. */
static int for_each_device(int (*visit)(Device, void *), void *context) {
    int seen = 0;
    for (size_t c = 0; c < sizeof kClasses / sizeof kClasses[0] && !seen; c++) {
        CFMutableDictionaryRef match = IOServiceMatching(kClasses[c]);
        io_iterator_t iterator = 0;
        if (!match || IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) != KERN_SUCCESS) {
            continue;
        }
        io_service_t service;
        int stop = 0;
        while (!stop && (service = IOIteratorNext(iterator))) {
            IOCFPlugInInterface **plugin = NULL;
            SInt32 score = 0;
            kern_return_t kr = IOCreatePlugInInterfaceForService(
                service, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &plugin, &score);
            IOObjectRelease(service);
            if (kr != KERN_SUCCESS || !plugin) {
                continue;
            }
            seen = 1;
            Device candidate = NULL;
            HRESULT result = (*plugin)->QueryInterface(
                plugin, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID500), (LPVOID *)&candidate);
            IODestroyPlugInInterface(plugin);
            if (result != S_OK || !candidate) {
                continue;
            }
            stop = visit(candidate, context);
        }
        IOObjectRelease(iterator);
        if (stop) {
            return 1;
        }
    }
    return 0;
}

/* Does configuration 0 contain a VideoControl interface (class 0x0E, subclass 0x01)? */
static int has_video_control(Device candidate) {
    IOUSBConfigurationDescriptorPtr config = NULL;
    if ((*candidate)->GetConfigurationDescriptorPtr(candidate, 0, &config) != kIOReturnSuccess || !config) {
        return 0;
    }
    const uint8_t *bytes = (const uint8_t *)config;
    uint16_t total = USBToHostWord(config->wTotalLength);
    for (uint16_t at = 0; at + 7 < total && bytes[at] >= 2; at += bytes[at]) {
        if (bytes[at + 1] == 0x04 && bytes[at + 5] == 0x0E && bytes[at + 6] == 0x01) {
            return 1;
        }
    }
    return 0;
}

struct list_context {
    uint32_t *ids;
    int capacity;
    int count;
};

static int list_visit(Device candidate, void *raw) {
    struct list_context *context = raw;
    if (has_video_control(candidate) && context->count < context->capacity) {
        UInt16 vendor = 0, product = 0;
        (*candidate)->GetDeviceVendor(candidate, &vendor);
        (*candidate)->GetDeviceProduct(candidate, &product);
        context->ids[context->count++] = ((uint32_t)vendor << 16) | product;
    }
    (*candidate)->Release(candidate);
    return 0;
}

/* Fills ids with (vendor << 16 | product) of every UVC device; returns how many. */
int trashdrop_uvc_list(uint32_t *ids, int capacity) {
    struct list_context context = {ids, capacity, 0};
    for_each_device(list_visit, &context);
    return context.count;
}

struct open_context {
    uint16_t vendor;
    uint16_t product;
    Device found;
};

static int open_visit(Device candidate, void *raw) {
    struct open_context *context = raw;
    UInt16 vendor = 0, product = 0;
    (*candidate)->GetDeviceVendor(candidate, &vendor);
    (*candidate)->GetDeviceProduct(candidate, &product);
    if (vendor == context->vendor && product == context->product && has_video_control(candidate)) {
        context->found = candidate;
        return 1;
    }
    (*candidate)->Release(candidate);
    return 0;
}

void trashdrop_uvc_close(void) {
    if (device) {
        if (device_is_open) {
            (*device)->USBDeviceClose(device);
        }
        (*device)->Release(device);
    }
    device = NULL;
    device_is_open = 0;
}

/* Opens the camera. Returns 0, -1 if it is not on the bus, or the IOReturn of
 * the open. A failed open keeps the handle: control requests on the default
 * pipe may still be allowed, and the caller finds out by trying one. */
int trashdrop_uvc_open(uint16_t vendor, uint16_t product) {
    trashdrop_uvc_close();
    struct open_context context = {vendor, product, NULL};
    for_each_device(open_visit, &context);
    if (!context.found) {
        return -1;
    }
    device = context.found;
    IOReturn kr = (*device)->USBDeviceOpen(device);
    if (kr == kIOReturnSuccess) {
        device_is_open = 1;
        return 0;
    }
    return (int)kr;
}

int trashdrop_uvc_config_descriptor(uint8_t *buffer, uint32_t capacity, uint32_t *length) {
    if (!device) {
        return -1;
    }
    IOUSBConfigurationDescriptorPtr config = NULL;
    IOReturn kr = (*device)->GetConfigurationDescriptorPtr(device, 0, &config);
    if (kr != kIOReturnSuccess || !config) {
        return (int)kr;
    }
    uint32_t total = USBToHostWord(config->wTotalLength);
    if (total > capacity) {
        total = capacity;
    }
    memcpy(buffer, config, total);
    *length = total;
    return 0;
}

int trashdrop_uvc_request(uint8_t request_type, uint8_t request, uint16_t value, uint16_t index,
                          uint8_t *data, uint16_t length, uint32_t *transferred) {
    if (!device) {
        return -1;
    }
    IOUSBDevRequestTO r;
    memset(&r, 0, sizeof r);
    r.bmRequestType = request_type;
    r.bRequest = request;
    r.wValue = value;
    r.wIndex = index;
    r.wLength = length;
    r.pData = data;
    r.noDataTimeout = 1000;
    r.completionTimeout = 1000;
    IOReturn kr = (*device)->DeviceRequestTO(device, &r);
    if (transferred) {
        *transferred = r.wLenDone;
    }
    return (int)kr;
}
