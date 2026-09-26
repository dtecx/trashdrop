"""The arms' servo buses: six Feetech STS3215 motors behind one USB adapter.

Each SO-101 has its six servos on a single half-duplex bus, reached through a
USB adapter that shows up as a serial port. The port's name depends on the USB
socket -- /dev/cu.usbmodem... on a Mac, COM5 on Windows -- but the adapter's
serial number does not, so that is how an arm is found.

This is the raw bus: registers in, registers out. Nothing here decides whether
a write is safe -- that is trashdrop/arm.py's job, and nothing else should
write to a servo.
"""

from __future__ import annotations

from dataclasses import dataclass

BAUD_RATE = 1_000_000
# Bus IDs as LeRobot assigns them on an SO-101.
MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
STS3215 = 777  # the model number an STS3215 answers a ping with
# Control table entries used here: name -> (address, bytes).
REGISTERS = {
    "min_limit": (9, 2),
    "max_limit": (11, 2),
    "homing_offset": (31, 2),
    "torque_enable": (40, 1),
    "acceleration": (41, 1),
    "goal_position": (42, 2),
    "goal_speed": (46, 2),
    "lock": (55, 1),  # 0: what is written to the EEPROM part of the table survives power off
    "position": (56, 2),
    "voltage": (62, 1),
    "temperature": (63, 1),
}


def list_buses() -> list[tuple[str, str]]:
    """(serial number, port) of every USB serial adapter plugged in."""

    from serial.tools import list_ports

    return [(port.serial_number, port.device) for port in list_ports.comports() if port.serial_number]


def find_port(serial_number: str) -> str:
    for serial, device in list_buses():
        if serial == serial_number:
            return device
    raise RuntimeError(f"no servo adapter with serial number {serial_number} -- is that arm plugged in?")


def decode_offset(raw: int) -> int:
    """Homing offset is sign-magnitude, with the sign in bit 11."""

    return -(raw & 0x7FF) if raw & 0x800 else raw


def encode_offset(value: int) -> int:
    """A homing offset as the servo stores it: sign-magnitude, the sign in bit 11."""

    if not -0x7FF <= value <= 0x7FF:
        raise ValueError(f"homing offset {value} is outside -2047..2047")
    return 0x800 | -value if value < 0 else value


@dataclass(frozen=True)
class MotorState:
    name: str
    motor: int
    position: int
    min_limit: int
    max_limit: int
    homing_offset: int
    torque: bool
    voltage: float
    temperature: int


class ServoBus:
    """One arm's bus, opened on a serial port."""

    def __init__(self, port: str) -> None:
        from scservo_sdk import PacketHandler, PortHandler

        self.port = port
        self._port = PortHandler(port)
        self._packet = PacketHandler(0)  # protocol 0 is the STS/SMS series
        if not self._port.openPort():
            raise RuntimeError(f"cannot open {port}")
        self._port.setBaudRate(BAUD_RATE)

    @classmethod
    def by_serial(cls, serial_number: str) -> "ServoBus":
        return cls(find_port(serial_number))

    def close(self) -> None:
        self._port.closePort()

    def __enter__(self) -> "ServoBus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def ping(self, motor: int) -> int | None:
        """The motor's model number, or None if nothing answers at that ID."""

        from scservo_sdk import COMM_SUCCESS

        model, result, _ = self._packet.ping(self._port, motor)
        return model if result == COMM_SUCCESS else None

    def read(self, motor: int, register: str) -> int:
        from scservo_sdk import COMM_SUCCESS

        address, size = REGISTERS[register]
        reader = self._packet.read1ByteTxRx if size == 1 else self._packet.read2ByteTxRx
        value, result, _ = reader(self._port, motor, address)
        if result != COMM_SUCCESS:
            raise RuntimeError(f"{self.port}: motor {motor} did not answer when reading {register}")
        return value

    def write(self, motor: int, register: str, value: int) -> None:
        from scservo_sdk import COMM_SUCCESS

        address, size = REGISTERS[register]
        writer = self._packet.write1ByteTxRx if size == 1 else self._packet.write2ByteTxRx
        result, _ = writer(self._port, motor, address, int(value))
        if result != COMM_SUCCESS:
            raise RuntimeError(f"{self.port}: motor {motor} did not acknowledge writing {register}")

    def write_goals(self, goals: dict[int, int]) -> None:
        """Goal positions for several motors in one broadcast packet (no replies)."""

        from scservo_sdk import GroupSyncWrite

        address, size = REGISTERS["goal_position"]
        group = GroupSyncWrite(self._port, self._packet, address, size)
        for motor, ticks in goals.items():
            value = int(ticks)
            group.addParam(motor, [value & 0xFF, (value >> 8) & 0xFF])
        group.txPacket()

    def missing(self) -> list[str]:
        """Joints whose motor does not answer, or is not an STS3215."""

        return [name for name, motor in MOTORS.items() if self.ping(motor) != STS3215]

    def positions(self) -> dict[str, int]:
        return {name: self.read(motor, "position") for name, motor in MOTORS.items()}

    def state(self) -> list[MotorState]:
        return [
            MotorState(
                name=name,
                motor=motor,
                position=self.read(motor, "position"),
                min_limit=self.read(motor, "min_limit"),
                max_limit=self.read(motor, "max_limit"),
                homing_offset=decode_offset(self.read(motor, "homing_offset")),
                torque=bool(self.read(motor, "torque_enable")),
                voltage=self.read(motor, "voltage") / 10,
                temperature=self.read(motor, "temperature"),
            )
            for name, motor in MOTORS.items()
        ]
