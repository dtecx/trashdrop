/**
 * HandStream: sends both hands from Spectacles to the TrashDrop Mac, where the
 * two SO-101 arms follow them (uv run python -m trashdrop.spectacles).
 *
 * Some 30 times a second it sends, as JSON over a WebSocket, the wearer's head
 * (position, and the camera's forward axis) and for each hand whether it is
 * tracked and where its wrist, thumb tip and index tip are -- world space,
 * centimetres. The Mac answers with what the arms are doing, shown in the
 * optional Status text.
 *
 * Setup: see spectacles/README.md. In short: a Spectacles project (it brings
 * the Spectacles Interaction Kit), an Internet Module asset, this script on
 * any scene object with its inputs set, and Experimental APIs on for ws://.
 */
import { SIK } from "SpectaclesInteractionKit.lspkg/SIK";

type Side = "left" | "right";

@component
export class HandStream extends BaseScriptComponent {
  @input
  @hint("The Internet Module asset (Asset Browser: + > Internet Module)")
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

  private socket: WebSocket | null = null;
  private open = false;
  private lastSent = 0;
  private retryAt = 0;

  onAwake() {
    this.createEvent("OnStartEvent").bind(() => this.connect());
    this.createEvent("UpdateEvent").bind(() => this.update());
  }

  private connect() {
    this.open = false;
    this.say("connecting to " + this.url);
    try {
      this.socket = this.internetModule.createWebSocket(this.url);
    } catch (error) {
      this.say("cannot connect: " + error + " (ws:// needs Experimental APIs on)");
      this.socket = null;
      this.retryAt = getTime() + 5;
      return;
    }
    this.socket.onopen = () => {
      this.open = true;
      this.say("connected");
    };
    this.socket.onmessage = (event: WebSocketMessageEvent) => {
      if (typeof event.data === "string") {
        this.say(event.data);
      }
    };
    this.socket.onclose = () => {
      this.open = false;
      this.socket = null;
      this.retryAt = getTime() + 2;
      this.say("disconnected: trying again");
    };
    this.socket.onerror = () => {
      this.say("connection error: is the Mac listening, on the same Wi-Fi?");
    };
  }

  private update() {
    const now = getTime();
    if (this.socket === null) {
      if (now >= this.retryAt) {
        this.connect();
      }
      return;
    }
    if (!this.open || now - this.lastSent < 1 / this.rate) {
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
    this.socket.send(JSON.stringify(message));
  }

  private say(text: string) {
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
    thumb: point(tracked.thumbTip.position),
    index: point(tracked.indexTip.position),
  };
}
