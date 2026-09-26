/** Show the overhead webcam on a head-locked Image in Spectacles (2024). */

const CONNECT_TIMEOUT_S = 5;
const RETRY_S = 2;

type VideoPacket = {
  seq: number;
  capturedMs: number;
  jpeg: string;
};

@component
export class WebcamView extends BaseScriptComponent {
  @input
  internetModule: InternetModule;

  @input
  image: Image;

  @input
  url: string = "ws://127.0.0.1:8766";

  @input
  previewToo: boolean = false;

  private socket: WebSocket | null = null;
  private open = false;
  private connectingSince = 0;
  private retryAt = 0;
  private decoding = false;
  private pending: VideoPacket | null = null;
  private shown = 0;
  private reportAt = 0;
  private totalAgeMs = 0;
  private agedFrames = 0;
  private clockOffsetMs: number | null = null;
  private syncRttMs = Infinity;
  private lastPingAt = 0;

  onAwake() {
    if (!this.image || !this.internetModule) {
      print("WebcamView: assign the Image and Internet Module in the Inspector");
      return;
    }
    if (global.deviceInfoSystem.isEditor() && !this.previewToo) {
      print("WebcamView: preview is not connecting (enable Preview Too to test on the Mac)");
      return;
    }
    this.createEvent("UpdateEvent").bind(() => this.update());
    this.createEvent("OnDestroyEvent").bind(() => this.drop(null));
  }

  private connect() {
    let socket: WebSocket;
    try {
      socket = this.internetModule.createWebSocket(this.url);
    } catch (error) {
      this.say("cannot connect: " + error);
      this.retryAt = getTime() + RETRY_S;
      return;
    }
    this.socket = socket;
    this.open = false;
    this.connectingSince = getTime();
    socket.onopen = () => {
      if (socket === this.socket) {
        this.open = true;
        this.say("connected");
        this.ping();
      }
    };
    socket.onmessage = (event: WebSocketMessageEvent) => {
      if (socket !== this.socket || typeof event.data !== "string") {
        return;
      }
      try {
        const packet = JSON.parse(event.data);
        if (typeof packet.pongMs === "number" && typeof packet.serverMs === "number") {
          const receivedMs = getTime() * 1000;
          const rttMs = receivedMs - packet.pongMs;
          if (rttMs >= 0 && rttMs < this.syncRttMs) {
            this.syncRttMs = rttMs;
            this.clockOffsetMs = packet.serverMs - (packet.pongMs + receivedMs) / 2;
          }
          return;
        }
        if (typeof packet.jpeg === "string" && typeof packet.capturedMs === "number") {
          this.pending = packet as VideoPacket; // Replace a waiting frame; never decode stale queued images.
          this.decodeLatest();
        }
      } catch (error) {
        this.say("invalid video packet: " + error);
      }
    };
    socket.onclose = () => {
      if (socket === this.socket) {
        this.drop("disconnected: trying again");
      }
    };
    socket.onerror = () => {
      if (socket === this.socket) {
        this.say("connection error: start the Mac bridge with --video");
      }
    };
  }

  private decodeLatest() {
    if (this.decoding || this.pending === null) {
      return;
    }
    const packet = this.pending;
    this.pending = null;
    this.decoding = true;
    Base64.decodeTextureAsync(packet.jpeg, (texture: Texture) => {
      this.image.mainPass.baseTex = texture;
      this.shown++;
      if (this.clockOffsetMs !== null) {
        this.totalAgeMs += getTime() * 1000 + this.clockOffsetMs - packet.capturedMs;
        this.agedFrames++;
      }
      const now = getTime();
      if (this.reportAt === 0) {
        this.reportAt = now;
      } else if (now - this.reportAt >= 1) {
        let age = "latency pending";
        if (this.agedFrames > 0) {
          const estimate = this.totalAgeMs / this.agedFrames;
          const uncertainty = this.syncRttMs / 2;
          age = Math.max(0, Math.round(estimate - uncertainty)) + "-" +
                Math.max(0, Math.round(estimate + uncertainty)) +
                " ms capture-to-display (clock RTT " + Math.round(this.syncRttMs) + " ms)";
        }
        this.say((this.shown / (now - this.reportAt)).toFixed(1) + " fps, " + age);
        this.shown = 0;
        this.totalAgeMs = 0;
        this.agedFrames = 0;
        this.reportAt = now;
      }
      this.decoding = false;
      this.decodeLatest();
    }, () => {
      this.decoding = false;
      this.say("JPEG decode failed");
      this.decodeLatest();
    });
  }

  private drop(why: string | null) {
    const socket = this.socket;
    this.socket = null;
    this.open = false;
    this.pending = null;
    this.clockOffsetMs = null;
    this.syncRttMs = Infinity;
    this.retryAt = getTime() + RETRY_S;
    if (why !== null) {
      this.say(why);
    }
    if (socket !== null) {
      try {
        socket.close();
      } catch (error) {
        // The socket can already be closed by the runtime.
      }
    }
  }

  private update() {
    if (this.socket === null) {
      if (getTime() >= this.retryAt) {
        this.connect();
      }
    } else if (!this.open && getTime() - this.connectingSince > CONNECT_TIMEOUT_S) {
      this.drop("no answer from " + this.url + ": trying again");
    } else if (this.open && getTime() - this.lastPingAt > 5) {
      this.ping();
    }
  }

  private ping() {
    if (this.socket === null || !this.open) {
      return;
    }
    this.lastPingAt = getTime();
    try {
      this.socket.send(JSON.stringify({ pingMs: this.lastPingAt * 1000 }));
    } catch (error) {
      this.drop("cannot sync clocks: " + error);
    }
  }

  private say(message: string) {
    print("WebcamView: " + message);
  }
}
