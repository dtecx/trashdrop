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
 * A plain ws:// address needs Project Settings > Allow Experimental API: such
 * a Lens runs on the developer's own glasses but cannot be published. Setup
 * and use: spectacles/README.md in the TrashDrop repository.
 */
import { SIK } from "SpectaclesInteractionKit.lspkg/SIK";

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
  @hint("Connect from Lens Studio's preview too, to try the link without the glasses. Leave it off otherwise: the preview runs on the Mac, reaches it at the same address, and its hands would mix with the glasses'")
  previewToo: boolean = false;

  private socket: WebSocket | null = null;
  private open = false;
  private connectingSince = 0;
  private lastSent = 0;
  private retryAt = 0;

  onAwake() {
    if (global.deviceInfoSystem.isEditor() && !this.previewToo) {
      // Mixed in, the preview's hands -- none, as a rule -- would re-anchor
      // each arm every other message, and the arms would barely move.
      this.say("Lens Studio's preview: not connecting (tick Preview Too to try the link here)");
      return;
    }
    this.createEvent("UpdateEvent").bind(() => this.update());
    this.createEvent("OnDestroyEvent").bind(() => this.drop(null));
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
        this.say(event.data);
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
    try {
      this.socket.send(JSON.stringify(message));
    } catch (error) {
      this.drop("cannot send: " + error);
    }
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
