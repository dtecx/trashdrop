/**
 * HandStream: sends both hands from Spectacles to the TrashDrop Mac, where the
 * two SO-101 arms follow them (uv run python -m trashdrop.spectacles).
 *
 * Some 30 times a second it sends, as JSON over a WebSocket, the wearer's head
 * (position, and the camera's forward axis) and for each hand whether it is
 * tracked and where its wrist, three knuckles and five fingertips are --
 * world space, centimetres. Knuckles give a stable hand frame while pinching;
 * the middle, ring and pinky tips tell a fist.
 * The Mac answers with what the arms are doing, shown in the optional Status text.
 *
 * A plain ws:// address needs Project Settings > Allow Experimental API: such
 * a Lens runs on the developer's own glasses but cannot be published. Setup
 * and use: spectacles/README.md in the TrashDrop repository.
 */
import { SIK } from "SpectaclesInteractionKit.lspkg/SIK";
import { SpectaclesUI } from "./SpectaclesUI";

type Side = "left" | "right";

// A refused connection closes at once; one to an address nobody answers on
// (the Mac's Wi-Fi address on a network that keeps devices apart) can hang
// far longer. Past this it is dropped and tried afresh.
const CONNECT_TIMEOUT_S = 5;
const RETRY_S = 2;

@component
export class HandStream extends BaseScriptComponent {
  @input
  @hint("The Internet Module asset")
  internetModule: InternetModule;

  @input
  @hint("The scene's Camera Object: where the wearer's head is and faces")
  camera: SceneObject;

  @input
  @hint("ws://127.0.0.1:8765 over the USB cable, or ws://<the Mac's address>:8765 over Wi-Fi -- the Mac prints it")
  url: string = "ws://127.0.0.1:8765";

  @input
  @hint("Messages a second")
  rate: number = 30;

  @input
  @allowUndefined
  @hint("Optional: a Text to show what the arms are doing")
  status: Text;

  @input
  @hint("The Render Target drawn by the scene camera; encoded when the Mac requests a snapshot")
  view: Texture;

  @input
  @allowUndefined
  @hint("The world-locked UIKit Frame containing the overhead video")
  videoFrame: SceneObject;

  @input
  @hint("Connect from Lens Studio's preview too, to try the link without the glasses. Leave it off otherwise: the preview runs on the Mac, reaches it at the same address, and its hands would mix with the glasses'")
  previewToo: boolean = false;

  private socket: WebSocket | null = null;
  private open = false;
  private connectingSince = 0;
  private lastSent = 0;
  private retryAt = 0;
  private cameraTexture: Texture | null = null;
  private textureEncoding = false;
  private captureReadyAt = 0;
  private snap: { id: number; kind: "snap" | "spectator"; view?: string; camera?: string } | null = null;
  private lastSnap: { id: number; view: string; camera: string } | null = null;
  private ui: SpectaclesUI;
  private lastSaid = "";

  onAwake() {
    this.ui = new SpectaclesUI(this.camera, (command: any) => this.sendCommand(command));
    if (this.videoFrame) {
      this.ui.placeVideo(this.videoFrame);
    }
    this.createEvent("OnDestroyEvent").bind(() => this.drop(null));
    if (global.deviceInfoSystem.isEditor() && !this.previewToo) {
      // Mixed in, the preview's hands -- none, as a rule -- would re-anchor
      // each arm every other message, and the arms would barely move.
      this.say("Lens Studio's preview: not connecting (tick Preview Too to try the link here)");
      return;
    }
    this.createEvent("UpdateEvent").bind(() => this.update());
    if (!global.deviceInfoSystem.isEditor()) {
      // Spectacles rejects createCameraRequest during onAwake. Defer it until
      // OnStart, after the Lens and its camera permissions are initialized.
      this.createEvent("OnStartEvent").bind(() => this.startCamera());
    }
  }

  private startCamera() {
    try {
      // CameraModule is @wearableOnly: even requiring it in the editor makes
      // the preview fail before it can show the rest of the Lens.
      const cameraModule = require("LensStudio:CameraModule") as CameraModule;
      const request = CameraModule.createCameraRequest();
      request.cameraId = CameraModule.CameraId.Default_Color;
      // The full 1392x1590 display Render Target crashes SnapOS readback on
      // this Spectacles build. A small raw camera frame is the documented
      // video-like source and is enough for the jury's spectator page.
      request.imageSmallerDimension = 360;
      this.cameraTexture = cameraModule.requestCamera(request);
      const provider = this.cameraTexture.control as CameraTextureProvider;
      // Snap's composite-streaming pattern starts texture readback only from a
      // real camera frame and lets the GPU fill its render target first.
      this.captureReadyAt = getTime() + 5;
      provider.onNewFrame.add(() => this.onCameraFrame());
    } catch (error) {
      // A missing optical camera must not take the hand link or control UI
      // down with it; snapshots stay unavailable while teleoperation works.
      this.cameraTexture = null;
      this.say("optical camera unavailable: " + error);
    }
  }

  private connect() {
    this.say("connecting to " + this.url);
    let socket: WebSocket;
    try {
      socket = this.internetModule.createWebSocket(this.url);
    } catch (error) {
      this.say("cannot connect to " + this.url + ": " + error);
      this.retryAt = getTime() + 5;
      return;
    }
    this.socket = socket;
    this.open = false;
    this.connectingSince = getTime();
    // Each handler checks it still belongs to the current socket: a socket
    // given up on can still report its close long after a new one opened.
    socket.onopen = () => {
      if (socket === this.socket) {
        this.open = true;
        this.say("connected");
      }
    };
    socket.onmessage = (event: WebSocketMessageEvent) => {
      if (socket === this.socket && typeof event.data === "string") {
        this.show(event.data);
      }
    };
    socket.onclose = () => {
      if (socket === this.socket) {
        this.drop("disconnected: trying again");
      }
    };
    socket.onerror = () => {
      if (socket === this.socket) {
        this.say("connection error: is the Mac side running? (uv run python -m trashdrop.spectacles)");
      }
    };
  }

  /** Lets the socket go and, with a reason to show, tries again shortly. */
  private drop(why: string | null) {
    const socket = this.socket;
    this.socket = null;
    this.open = false;
    this.retryAt = getTime() + RETRY_S;
    if (why !== null) {
      this.say(why);
    }
    if (socket !== null) {
      try {
        socket.close();
      } catch (error) {
        // already closed
      }
    }
  }

  private update() {
    const now = getTime();
    if (this.socket === null) {
      if (now >= this.retryAt) {
        this.connect();
      }
      return;
    }
    if (!this.open) {
      if (now - this.connectingSince > CONNECT_TIMEOUT_S) {
        this.drop("no answer from " + this.url + ": trying again");
      }
      return;
    }
    if (now - this.lastSent < 1 / this.rate) {
      return;
    }
    this.lastSent = now;
    const head = this.camera.getTransform();
    const message = {
      t: now,
      // Either way along the camera's axis will do: the Mac turns it to face the hands.
      head: { p: point(head.getWorldPosition()), look: point(head.forward) },
      left: hand("left"),
      right: hand("right"),
    };
    if (this.ui.updateHands(message)) {
      // A pinch on the wrist menu is an interface action. The follower's
      // existing hand-loss behavior holds both arms until the menu closes.
      message.left = { tracked: false };
      message.right = { tracked: false };
    }
    try {
      this.socket.send(JSON.stringify(message));
    } catch (error) {
      this.drop("cannot send: " + error);
    }
  }

  /**
   * What the Mac answers: JSON with the status and, per hand, its box, what
   * it pushes on in vain and the finger's height above the table (see
   * trashdrop/spectacles.py, Follower.guide) -- or, from an older bridge,
   * plain words. The camera looks straight down on the table, so how high
   * the jaw is shows only here; at the table, the text turns red.
   */
  private show(data: string) {
    let report: any = null;
    try {
      report = JSON.parse(data);
    } catch (error) {
      report = null;
    }
    if (report !== null && typeof report === "object" && typeof report.snap === "number") {
      this.capture(report.snap, "snap");
    }
    if (report !== null && typeof report === "object" && typeof report.spectator === "number") {
      this.capture(report.spectator, "spectator");
    }
    if (report !== null && typeof report === "object") {
      this.ui.updateReport(report);
    }
    if (report === null || typeof report !== "object" || typeof report.status !== "string") {
      this.say(data);
      return;
    }
    const hands = report.hands || {};
    const tips: string[] = [];
    let atTable = false;
    for (const side of ["left", "right"]) {
      const guide = hands[side];
      if (!guide) {
        continue;
      }
      if (typeof guide.tip === "number") {
        tips.push(side + " tip " + guide.tip + " cm");
      }
      if (Array.isArray(guide.blocked) && guide.blocked.indexOf("down") >= 0) {
        atTable = true;
      }
    }
    this.say(report.status + (tips.length > 0 ? "\n" + tips.join(" | ") : ""));
    if (this.status) {
      this.status.textFill.color = atTable ? new vec4(1, 0.25, 0.2, 1) : new vec4(1, 1, 1, 1);
    }
  }

  private capture(id: number, kind: "snap" | "spectator") {
    if (kind === "snap" && this.lastSnap !== null && this.lastSnap.id === id) {
      this.sendCapture({ ...this.lastSnap, kind: "snap" });
      return;
    }
    if (this.snap !== null) {
      return;
    }
    if (this.cameraTexture === null || (kind === "snap" && !this.view)) {
      this.say("snapshot unavailable: run the Lens on Spectacles and assign its Render Target");
      return;
    }
    this.snap = { id: id, kind: kind };
    this.textureEncoding = false;
    if (kind === "spectator") {
      print("HandStream: queued spectator camera " + id);
      return;
    }
    this.textureEncoding = true;
    Base64.encodeTextureAsync(this.view, (jpeg: string) => {
      this.textureEncoding = false;
      if (this.snap !== null && this.snap.id === id) {
        this.snap.view = jpeg;
      }
    }, () => {
      this.textureEncoding = false;
      this.failSnapshot(id, "cannot encode the rendered view");
    },
    CompressionQuality.HighQuality, EncodingType.Jpg);
    // The colour camera is deliberately encoded only after the render target
    // finishes. Starting both jobs together -- and another camera job on every
    // 30 fps frame until the first callback -- can make SnapOS kill the Lens.
  }

  private onCameraFrame() {
    if (this.snap !== null && this.snap.kind === "spectator") {
      this.encodeSpectatorFrame();
    } else {
      this.encodeCameraFrame();
    }
  }

  private encodeSpectatorFrame() {
    if (this.snap === null || this.snap.kind !== "spectator" || this.textureEncoding ||
        getTime() < this.captureReadyAt || this.cameraTexture === null || this.cameraTexture.getWidth() <= 0) {
      return;
    }
    const id = this.snap.id;
    this.textureEncoding = true;
    print("HandStream: encoding spectator camera " + id + " at " +
      this.cameraTexture.getWidth() + "x" + this.cameraTexture.getHeight());
    Base64.encodeTextureAsync(this.cameraTexture, (jpeg: string) => {
      this.textureEncoding = false;
      if (this.snap !== null && this.snap.id === id) {
        this.snap = null;
        this.sendSpectator(id, jpeg);
      }
    }, () => {
      this.textureEncoding = false;
      this.failSnapshot(id, "cannot encode the spectator camera");
    }, CompressionQuality.LowQuality, EncodingType.Jpg);
  }

  private encodeCameraFrame() {
    if (this.snap === null || this.snap.kind !== "snap" || this.snap.view === undefined || this.snap.camera !== undefined ||
        this.textureEncoding || this.cameraTexture === null) {
      return;
    }
    const id = this.snap.id;
    this.textureEncoding = true;
    Base64.encodeTextureAsync(this.cameraTexture, (jpeg: string) => {
      this.textureEncoding = false;
      if (this.snap !== null && this.snap.id === id) {
        this.snap.camera = jpeg;
        this.finishSnapshot();
      }
    }, () => {
      this.textureEncoding = false;
      this.failSnapshot(id, "cannot encode the colour camera");
    }, CompressionQuality.HighQuality, EncodingType.Jpg);
  }

  private finishSnapshot() {
    if (this.snap === null || this.snap.view === undefined || this.snap.camera === undefined) {
      return;
    }
    const ready = { id: this.snap.id, kind: this.snap.kind, view: this.snap.view, camera: this.snap.camera };
    this.snap = null;
    this.textureEncoding = false;
    if (ready.kind === "snap") {
      this.lastSnap = ready;
    }
    this.sendCapture(ready);
  }

  private sendCapture(ready: { id: number; kind: "snap" | "spectator"; view: string; camera: string }) {
    if (this.socket === null || !this.open) {
      return;
    }
    try {
      const message: any = { view: ready.view, camera: ready.camera };
      message[ready.kind] = ready.id;
      this.socket.send(JSON.stringify(message));
      if (ready.kind === "snap") {
        print("HandStream: sent snapshot " + ready.id);
      }
    } catch (error) {
      this.drop("cannot send snapshot: " + error);
    }
  }

  private sendSpectator(id: number, composite: string) {
    if (this.socket === null || !this.open) {
      return;
    }
    try {
      this.socket.send(JSON.stringify({ spectator: id, composite: composite }));
      print("HandStream: sent spectator composite " + id + " (" + Math.round(composite.length / 1024) + " KiB)");
    } catch (error) {
      this.drop("cannot send spectator view: " + error);
    }
  }

  private failSnapshot(id: number, reason: string) {
    if (this.snap !== null && this.snap.id === id) {
      this.snap = null;
    }
    this.textureEncoding = false;
    this.say("snapshot " + id + ": " + reason);
  }

  private sendCommand(command: any) {
    if (this.socket === null || !this.open) {
      this.say("control unavailable: the Mac bridge is not connected");
      return;
    }
    try {
      this.socket.send(JSON.stringify(command));
    } catch (error) {
      this.drop("cannot send control: " + error);
    }
  }

  private say(text: string) {
    if (text === this.lastSaid) {
      return;
    }
    this.lastSaid = text;
    print("HandStream: " + text);
    if (this.status) {
      this.status.text = text;
    }
  }
}

function point(p: vec3): number[] {
  return [Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10, Math.round(p.z * 10) / 10];
}

function hand(side: Side) {
  const tracked = SIK.HandInputData.getHand(side);
  if (!tracked.isTracked()) {
    return { tracked: false };
  }
  return {
    tracked: true,
    wrist: point(tracked.wrist.position),
    middleKnuckle: point(tracked.middleKnuckle.position),
    indexKnuckle: point(tracked.indexKnuckle.position),
    pinkyKnuckle: point(tracked.pinkyKnuckle.position),
    thumb: point(tracked.thumbTip.position),
    index: point(tracked.indexTip.position),
    // Folded back towards the wrist, these three make a fist: the Mac turns the jaw with it.
    middleTip: point(tracked.middleTip.position),
    ringTip: point(tracked.ringTip.position),
    pinkyTip: point(tracked.pinkyTip.position),
    // The glasses' own pinch detection: steadier than thumb-to-index distance alone.
    pinch: tracked.isPinching(),
  };
}
