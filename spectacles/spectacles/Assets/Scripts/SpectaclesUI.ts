/** World-locked Spectacles UI for the two TrashDrop arms. */

import {CapsuleButton} from "SpectaclesUIKit.lspkg/Scripts/Components/Button/CapsuleButton";

type Command = { command: string; [key: string]: unknown };

type ArmGuide = {
  mode?: string;
  state?: string;
  blocked?: string[];
  tip?: number;
  jaw?: number;
  roll?: number;
  turn?: number;
};

type Card = {
  object: SceneObject;
  text: Text;
  marker: Text;
  guide: ArmGuide | null;
};

const OPEN_PERCENT = 30;
const HEIGHT_RANGE = [3, 35];

export class SpectaclesUI {
  private cards: { left: Card; right: Card };
  private view: Camera;
  private precision = false;
  private jawOpen = false;
  private precisionLabel: Text;
  private gripLabel: Text;
  private stopped = false;
  private menu: SceneObject;
  private exitButton: SceneObject;
  private video: SceneObject | null = null;
  private presentationVisible = true;

  constructor(private camera: SceneObject, private send: (command: Command) => void) {
    this.view = camera.getComponent("Component.Camera") as Camera;
    this.cards = {
      left: this.createCard("left", new vec2(0.27, 0.18)),
      right: this.createCard("right", new vec2(0.73, 0.18)),
    };
    this.menu = this.createMenu();
    this.renderCard("left", null);
    this.renderCard("right", null);
  }

  public placeVideo(frame: SceneObject) {
    this.video = frame;
    this.place(frame, new vec2(0.5, 0.55), 62);
  }

  public updateReport(report: any) {
    if (typeof report.presentation === "boolean") {
      this.setPresentation(report.presentation);
    }
    const arms = report && typeof report === "object" ? report.arms || report.hands || {} : {};
    for (const side of ["left", "right"] as const) {
      const guide = arms[side] && typeof arms[side] === "object" ? arms[side] as ArmGuide : null;
      this.cards[side].guide = guide;
      this.renderCard(side, guide);
    }
    this.precision = typeof report.scale === "number" && report.scale <= 0.5;
    this.stopped = report.stopped === true;
    this.setText(this.precisionLabel, this.precision ? "PRECISE  1/2" : "PRECISION");
    const jaws = [this.cards.left.guide, this.cards.right.guide]
      .filter((guide) => guide !== null && typeof guide.jaw === "number") as ArmGuide[];
    this.jawOpen = jaws.length > 0 && jaws.every((guide) => (guide.jaw as number) >= OPEN_PERCENT);
    this.setText(this.gripLabel, this.jawOpen ? "CLOSE GRIPS" : "OPEN GRIPS");
  }

  public updateHands(message: any) {
    for (const side of ["left", "right"] as const) {
      const hand = message && message[side];
      const marker = this.cards[side].marker;
      if (!this.presentationVisible || !hand || !hand.tracked || !Array.isArray(hand.wrist)) {
        marker.getSceneObject().enabled = false;
        continue;
      }
      marker.getSceneObject().enabled = true;
      marker.getTransform().setWorldPosition(new vec3(hand.wrist[0], hand.wrist[1] + 4, hand.wrist[2]));
      marker.getTransform().setWorldRotation(this.camera.getTransform().getWorldRotation());
      marker.textFill.color = this.colour(this.cards[side].guide);
    }
  }

  private createCard(side: "left" | "right", screen: vec2): Card {
    const object = global.scene.createSceneObject((side === "left" ? "Left" : "Right") + " Arm Card");
    this.place(object, screen, 62);
    const text = this.createText(object, 14, 11, 22);
    text.backgroundSettings.enabled = true;
    text.backgroundSettings.cornerRadius = 0.22;
    text.backgroundSettings.margins = Rect.create(1, 1, 1, 1);

    const markerObject = global.scene.createSceneObject((side === "left" ? "Left" : "Right") + " Hand Label");
    const marker = this.createText(markerObject, 5, 5, 34);
    marker.text = side === "left" ? "L" : "R";
    marker.backgroundSettings.enabled = true;
    marker.backgroundSettings.cornerRadius = 0.5;
    marker.backgroundSettings.margins = Rect.create(0.6, 0.6, 0.6, 0.6);
    markerObject.enabled = false;
    return { object: object, text: text, marker: marker, guide: null };
  }

  private createMenu(): SceneObject {
    const menu = global.scene.createSceneObject("Arm Control Menu");
    this.place(menu, new vec2(0.5, 0.82), 62);
    const grip = this.createButton(menu, "OPEN GRIPS", -7.6, 2.1, () => {
      this.send({ command: "jaw", open: !this.jawOpen });
    });
    this.gripLabel = grip.label;
    const precision = this.createButton(menu, "PRECISION", 0, 2.1, () => {
      this.precision = !this.precision;
      this.setText(this.precisionLabel, this.precision ? "PRECISE  1/2" : "PRECISION");
      this.send({ command: "precision", enabled: this.precision });
    });
    this.precisionLabel = precision.label;
    this.createButton(menu, "HOME", 7.6, 2.1, () => this.send({ command: "home" }));
    const stop = this.createButton(menu, "STOP", -3.8, -2.1, () => {
      if (!this.stopped) {
        this.send({ command: "stop" });
      }
    });
    stop.label.textFill.color = new vec4(1, 0.3, 0.25, 1);
    const exit = this.createButton(menu, "WEB UI", 3.8, -2.1, () => {
      this.send({ command: "presentation", enabled: false });
    });
    this.exitButton = exit.object;
    this.exitButton.enabled = false;
    return menu;
  }

  private createButton(parent: SceneObject, labelText: string, x: number, y: number, action: () => void) {
    const object = global.scene.createSceneObject(labelText);
    object.setParent(parent);
    object.getTransform().setLocalPosition(new vec3(x, y, 0));
    const button = object.createComponent(CapsuleButton.getTypeName()) as CapsuleButton;
    button.size = new vec3(7, 3.6, 1);
    button.playAudio = false;
    button.onTriggerUp.add(action);
    const labelObject = global.scene.createSceneObject(labelText + " Label");
    labelObject.setParent(object);
    // Put labels in front of the UIKit mesh. Coplanar text flickers on device.
    labelObject.getTransform().setLocalPosition(new vec3(0, 0, 0.55));
    const label = this.createText(labelObject, 6.6, 3, 18);
    label.text = labelText;
    return { object: object, button: button, label: label };
  }

  private setPresentation(visible: boolean) {
    if (visible === this.presentationVisible) {
      return;
    }
    this.presentationVisible = visible;
    for (const side of ["left", "right"] as const) {
      this.cards[side].object.enabled = visible;
      if (!visible) {
        this.cards[side].marker.getSceneObject().enabled = false;
      }
    }
    this.menu.enabled = visible;
    this.exitButton.enabled = visible;
    if (this.video !== null) {
      this.video.enabled = visible;
    }
  }

  private createText(object: SceneObject, width: number, height: number, size: number): Text {
    const text = object.createComponent("Component.Text") as Text;
    text.text = "";
    text.size = size;
    text.sizeToFit = false;
    text.horizontalAlignment = HorizontalAlignment.Center;
    text.verticalAlignment = VerticalAlignment.Center;
    text.horizontalOverflow = HorizontalOverflow.Shrink;
    text.verticalOverflow = VerticalOverflow.Shrink;
    text.worldSpaceRect = Rect.create(-width / 2, width / 2, -height / 2, height / 2);
    text.textFill.color = new vec4(1, 1, 1, 1);
    text.depthTest = false;
    text.twoSided = false;
    return text;
  }

  private setText(text: Text, value: string) {
    if (text.text !== value) {
      text.text = value;
    }
  }

  private place(object: SceneObject, screen: vec2, distance: number) {
    const head = this.camera.getTransform();
    object.getTransform().setWorldPosition(this.view.screenSpaceToWorldSpace(screen, distance));
    object.getTransform().setWorldRotation(head.getWorldRotation());
  }

  private renderCard(side: "left" | "right", guide: ArmGuide | null) {
    const card = this.cards[side];
    const colour = this.colour(guide);
    const title = side === "left" ? "LEFT ARM" : "RIGHT ARM";
    let body: string;
    if (guide === null) {
      body = title + "\nOFFLINE\nGRIP  --\nHEIGHT  --------\nTURN  --";
    } else {
      const state = this.stateName(guide);
      const jaw = typeof guide.jaw === "number" && guide.jaw >= OPEN_PERCENT ? "OPEN" : "CLOSED";
      const tip = typeof guide.tip === "number" ? guide.tip : 0;
      const fill = Math.max(0, Math.min(8, Math.round((tip - HEIGHT_RANGE[0]) /
        (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) * 8)));
      const bar = "=".repeat(fill) + "-".repeat(8 - fill);
      const turn = typeof guide.turn === "number" ? guide.turn : (typeof guide.roll === "number" ? guide.roll : 0);
      const reason = guide.state && guide.state.indexOf(": at ") >= 0 ?
        "\n" + guide.state.split(": at ")[1].toUpperCase() : "";
      body = title + "\n" + state + reason + "\nGRIP  " + jaw +
        "\nHEIGHT  " + bar + "  " + Math.round(tip) + " cm\nTURN  " +
        (turn > 0 ? "+" : "") + Math.round(turn) + " deg";
    }
    this.setText(card.text, body);
    card.text.textFill.color = colour;
    card.text.backgroundSettings.fill.color = new vec4(colour.x * 0.18, colour.y * 0.18, colour.z * 0.18, 0.88);
    card.marker.backgroundSettings.fill.color = new vec4(colour.x * 0.2, colour.y * 0.2, colour.z * 0.2, 0.9);
  }

  private stateName(guide: ArmGuide): string {
    if (guide.blocked && guide.blocked.length > 0) {
      return "BLOCKED";
    }
    if (guide.state === "stopped") return "STOPPED";
    if (guide.state && guide.state.indexOf("returning home") === 0) return "HOMING";
    if (guide.state && guide.state.indexOf("waiting while") === 0) return "WAITING";
    const mode = guide.mode || "holding";
    if (mode === "moving") return "DRAGGING";
    if (mode === "turning") return "TURNING";
    if (mode === "lost") return "HAND LOST";
    if (mode === "calibrating") return "CALIBRATING";
    return "FREE";
  }

  private colour(guide: ArmGuide | null): vec4 {
    if (guide === null) return new vec4(0.55, 0.58, 0.65, 1);
    if (guide.state === "stopped") return new vec4(1, 0.18, 0.12, 1);
    if (guide.blocked && guide.blocked.length > 0) return new vec4(1, 0.35, 0.12, 1);
    if (guide.mode === "moving") return new vec4(0.15, 0.8, 1, 1);
    if (guide.mode === "turning") return new vec4(0.75, 0.4, 1, 1);
    if (guide.mode === "lost" || guide.mode === "calibrating") return new vec4(0.65, 0.68, 0.75, 1);
    return new vec4(0.25, 1, 0.5, 1);
  }
}
