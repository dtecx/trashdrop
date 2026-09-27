/** Spatial Spectacles UI: a movable video Frame, hand tags, and a wrist menu. */

import {CapsuleButton} from "SpectaclesUIKit.lspkg/Scripts/Components/Button/CapsuleButton";
import {Frame} from "SpectaclesUIKit.lspkg/Scripts/Components/Frame/Frame";

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

type Side = "left" | "right";
type HandTag = { text: Text; guide: ArmGuide | null; colourKey: string };
// Keep the control HUD deterministically above the movable video Frame.
// Shared render order zero flickers as coplanar UIKit surfaces rotate in stereo.
const HUD_RENDER_ORDER = 10;
// UIKit may still own a hovered button after trigger-up. Park the live menu
// outside the view instead of disabling its hierarchy in a hover callback.
const HIDE_DELAY_S = 1;
// A deliberate look at a raised wrist reveals the menu, not a passing glance.
const MENU_DWELL_S = 0.45;
const MENU_HIDE_S = 1.6;
const MENU_GAZE_COSINE = 0.87;

export class SpectaclesUI {
  private tags: { left: HandTag; right: HandTag };
  private view: Camera;
  private manualActive = false;
  private autoActive = false;
  private manualHighlight: boolean | null = null;
  private autoHighlight: boolean | null = null;
  private manualLabel: Text;
  private autoLabel: Text;
  private modeLabel: Text;
  private menu: SceneObject;
  private menuVisible = false;
  private menuSide: Side = "left";
  private gazeSide: Side | null = null;
  private gazeSince = 0;
  private lastMenuLook = 0;
  private lastMenuPinch = 0;
  private menuCloseAt = 0;
  private menuNeedsLookAway = false;
  private emptyButton: SceneObject;
  private emptyNeeded: boolean | null = null;
  private video: SceneObject | null = null;
  private frameEventsBound = false;
  private frameInteracting = false;
  private frameInteractionAt = 0;
  private uiNeedsRelease = false;
  private uiReleaseAfter = 0;
  private presentationVisible = true;
  private hideAt = 0;

  constructor(private camera: SceneObject, private send: (command: Command) => void) {
    this.view = camera.getComponent("Component.Camera") as Camera;
    this.tags = { left: this.createTag("left"), right: this.createTag("right") };
    this.menu = this.createMenu();
    this.parkMenu();
  }

  public placeVideo(frame: SceneObject) {
    this.video = frame;
    this.place(frame, new vec2(0.5, 0.55), 78);
  }

  public updateReport(report: any) {
    this.applyDelayedHide();
    if (typeof report.presentation === "boolean") {
      this.setPresentation(report.presentation);
    }
    if (typeof report.emptyPhotographed === "boolean") {
      const needed = !report.emptyPhotographed;
      if (needed !== this.emptyNeeded) {
        this.emptyNeeded = needed;
        if (!this.menuVisible) this.positionEmptyButton();
      }
    }
    if (typeof report.controlError === "string" && report.controlError) {
      this.menuCloseAt = 0;
    }
    this.manualActive = report.manual === true;
    this.autoActive = report.auto === true;
    const arms = report.manual ? report.arms || report.hands || {} : {};
    for (const side of ["left", "right"] as const) {
      this.tags[side].guide = arms[side] && typeof arms[side] === "object" ? arms[side] as ArmGuide : null;
    }
    if (this.manualHighlight !== this.manualActive) {
      this.highlight(this.manualLabel, this.manualActive, new vec4(0.15, 0.8, 1, 1));
      this.manualHighlight = this.manualActive;
    }
    if (this.autoHighlight !== this.autoActive) {
      this.highlight(this.autoLabel, this.autoActive, new vec4(0.3, 1, 0.5, 1));
      this.autoHighlight = this.autoActive;
    }
    const error = typeof report.controlError === "string" ? report.controlError : "";
    const mode = error === "photograph the empty zone first" ? "CLEAR ZONE · TAP EMPTY" : error ||
      (report.busy === "empty" ? "CAPTURING EMPTY ZONE" :
       this.manualActive ? "MANUAL · HAND CONTROL" : this.autoActive ? "AUTO · SORTING" :
       report.busy === "neutral" ? "MOVING TO NEUTRAL" :
       this.emptyNeeded ? "CLEAR ZONE · TAP EMPTY" : "EMPTY READY · TAP AUTO");
    this.setText(this.modeLabel, mode.toUpperCase());
  }

  /** Return true while a menu pinch must not drive the robot. */
  public updateHands(message: any): boolean {
    this.applyDelayedHide();
    this.bindFrameInteractions();
    this.updateMenu(message);
    const now = getTime();
    const pinching = !!((message.left && message.left.pinch) || (message.right && message.right.pinch));
    // A missed UIKit end event must not leave manual control held forever.
    if (this.frameInteracting && !pinching && now - this.frameInteractionAt > 0.2) {
      this.endFrameInteraction();
    }
    if (this.uiNeedsRelease && !this.frameInteracting && !pinching && now >= this.uiReleaseAfter) {
      this.uiNeedsRelease = false;
    }
    for (const side of ["left", "right"] as const) {
      const hand = message && message[side];
      const tag = this.tags[side];
      const object = tag.text.getSceneObject();
      if (!this.presentationVisible || !this.manualActive || !this.validWrist(hand)) {
        if (object.enabled) object.enabled = false;
        continue;
      }
      if (!object.enabled) object.enabled = true;
      object.getTransform().setWorldPosition(new vec3(hand.wrist[0], hand.wrist[1] + 5, hand.wrist[2]));
      object.getTransform().setWorldRotation(this.camera.getTransform().getWorldRotation());
      const colourKey = JSON.stringify([tag.guide && tag.guide.state, tag.guide && tag.guide.mode,
        tag.guide && tag.guide.blocked]);
      if (colourKey !== tag.colourKey) {
        const colour = this.colour(tag.guide);
        tag.text.textFill.color = colour;
        tag.text.backgroundSettings.fill.color = new vec4(colour.x * 0.12, colour.y * 0.12, colour.z * 0.12, 0.88);
        tag.colourKey = colourKey;
      }
      this.setText(tag.text, this.frameInteracting || this.uiNeedsRelease ?
        "UI CONTROL\nRELEASE TO DRIVE" :
        this.menuVisible ? "MENU OPEN\nLOOK AWAY TO DRIVE" : this.tagText(side, tag.guide));
    }
    // The bridge holds an arm when its tracked hand is temporarily missing.
    return (this.presentationVisible && (this.menuVisible || this.frameInteracting || this.uiNeedsRelease)) ||
      this.hideAt > 0;
  }

  private createTag(side: Side): HandTag {
    const object = global.scene.createSceneObject((side === "left" ? "Left" : "Right") + " Hand Tag");
    const text = this.createText(object, 10, 3.8, 17);
    text.backgroundSettings.enabled = true;
    text.backgroundSettings.cornerRadius = 0.28;
    text.backgroundSettings.margins = Rect.create(0.7, 0.7, 0.4, 0.4);
    object.enabled = false;
    return { text: text, guide: null, colourKey: "" };
  }

  private createMenu(): SceneObject {
    const menu = global.scene.createSceneObject("Wrist Control Menu");
    this.place(menu, new vec2(0.5, 0.7), 55);
    const modeObject = global.scene.createSceneObject("Control Mode");
    modeObject.setParent(menu);
    modeObject.getTransform().setLocalPosition(new vec3(0, 5, 0));
    this.modeLabel = this.createText(modeObject, 17, 2.3, 17);
    this.modeLabel.text = "READY";
    this.manualLabel = this.createButton(menu, "MANUAL", -4.3, 1.5, 8, 4, 22, () => {
      this.send({ command: "manual", enabled: !this.manualActive });
      this.queueMenuClose();
    });
    this.autoLabel = this.createButton(menu, "AUTO", 4.3, 1.5, 8, 4, 22, () => {
      this.send({ command: "auto", enabled: !this.autoActive });
      this.queueMenuClose();
    });
    const neutral = this.createButton(menu, "NEUTRAL", -1.8, -3, 10, 3.5, 20,
      () => {
        this.send({ command: "neutral" });
        this.queueMenuClose();
      });
    neutral.textFill.color = new vec4(1, 0.82, 0.35, 1);
    const exit = this.createButton(menu, "EXIT UI", 6, -3, 5, 3.5, 16, () => {
      this.send({ command: "presentation", enabled: false });
    });
    exit.textFill.color = new vec4(0.72, 0.75, 0.82, 1);
    const empty = this.createButton(menu, "EMPTY ZONE", 0, -7.2, 11, 3.5, 18,
      () => this.send({ command: "empty" }));
    empty.textFill.color = new vec4(1, 0.82, 0.35, 1);
    this.emptyButton = empty.getSceneObject().getParent();
    this.positionEmptyButton();
    return menu;
  }

  private createButton(parent: SceneObject, words: string, x: number, y: number,
                       width: number, height: number, fontSize: number, action: () => void): Text {
    const object = global.scene.createSceneObject(words + " Button");
    object.setParent(parent);
    object.getTransform().setLocalPosition(new vec3(x, y, 0));
    const button = object.createComponent(CapsuleButton.getTypeName()) as CapsuleButton;
    button.size = new vec3(width, height, 1);
    button.renderOrder = HUD_RENDER_ORDER;
    button.playAudio = false;
    button.onTriggerUp.add(action);
    const labelObject = global.scene.createSceneObject(words + " Label");
    labelObject.setParent(object);
    // Render order, not geometric separation, keeps stereo text stable.
    labelObject.getTransform().setLocalPosition(new vec3(0, 0, 0.01));
    const label = this.createText(labelObject, width - 0.4, height - 0.3, fontSize);
    label.text = words;
    label.backgroundSettings.enabled = true;
    label.backgroundSettings.cornerRadius = 0.35;
    label.backgroundSettings.fill.color = new vec4(0.08, 0.11, 0.17, 0.08);
    return label;
  }

  private highlight(label: Text, active: boolean, colour: vec4) {
    label.textFill.color = active ? colour : new vec4(0.9, 0.92, 0.95, 1);
    label.backgroundSettings.fill.color = active ?
      new vec4(colour.x * 0.2, colour.y * 0.2, colour.z * 0.2, 0.85) :
      new vec4(0.08, 0.11, 0.17, 0.08);
  }

  private bindFrameInteractions() {
    if (this.video === null || this.frameEventsBound) return;
    const frame = this.video.getComponent(Frame.getTypeName()) as Frame;
    if (!frame) return;
    // The Frame's center is cut out: a pinch through the video remains a
    // teleop gesture. Only an actual border drag or resize clutches the arms.
    const start = () => {
      this.frameInteracting = true;
      this.frameInteractionAt = getTime();
      this.uiNeedsRelease = true;
    };
    frame.onTranslationStart.add(start);
    frame.onScalingStart.add(start);
    frame.onTranslationEnd.add(() => this.endFrameInteraction());
    frame.onScalingEnd.add(() => this.endFrameInteraction());
    this.frameEventsBound = true;
  }

  private endFrameInteraction() {
    this.frameInteracting = false;
    this.uiReleaseAfter = getTime() + 0.35;
  }

  private queueMenuClose() {
    // Keep UIKit alive through trigger-up, then require a fresh look to reopen.
    this.menuCloseAt = getTime() + HIDE_DELAY_S;
  }

  private positionEmptyButton() {
    this.emptyButton.getTransform().setLocalPosition(new vec3(0, this.emptyNeeded ? -7.2 : -10000, 0));
  }

  private updateMenu(message: any) {
    if (!this.presentationVisible) {
      this.gazeSide = null;
      return;
    }
    const now = getTime();
    const lookedAt = this.lookedAtRaisedHand(message);
    if (this.menuNeedsLookAway) {
      if (lookedAt === null && !this.lookedAtMenu(message.head)) {
        this.menuNeedsLookAway = false;
        this.gazeSide = null;
        this.gazeSince = now;
      }
      return;
    }
    if (lookedAt !== this.gazeSide) {
      this.gazeSide = lookedAt;
      this.gazeSince = now;
    }
    if (!this.menuVisible && lookedAt !== null && now - this.gazeSince >= MENU_DWELL_S) {
      this.menuSide = lookedAt;
      this.positionEmptyButton();
      this.placeMenuByWrist(message[lookedAt].wrist, message.head);
      this.menuVisible = true;
      this.lastMenuLook = now;
    }
    if (!this.menuVisible) return;
    if (lookedAt === this.menuSide || this.lookedAtMenu(message.head)) this.lastMenuLook = now;
    const pinching = !!((message.left && message.left.pinch) || (message.right && message.right.pinch));
    if (pinching) this.lastMenuPinch = now;
    if (this.menuCloseAt > 0 && now >= this.menuCloseAt && !pinching) {
      this.menuCloseAt = 0;
      this.menuVisible = false;
      this.menuNeedsLookAway = true;
      this.uiNeedsRelease = true;
      this.uiReleaseAfter = now + 0.35;
      this.parkMenu();
      return;
    }
    if (!pinching && now - Math.max(this.lastMenuLook, this.lastMenuPinch) > MENU_HIDE_S) {
      this.menuVisible = false;
      this.parkMenu();
    }
  }

  private lookedAtRaisedHand(message: any): Side | null {
    const head = message && message.head;
    if (!head || !Array.isArray(head.p)) return null;
    for (const side of ["left", "right"] as const) {
      const hand = message[side];
      if (!this.validWrist(hand)) continue;
      const wrist = hand.wrist;
      // At the table a wrist is much lower than the head; it must be raised.
      if (wrist[1] < head.p[1] - 45) continue;
      const dx = wrist[0] - head.p[0];
      const dy = wrist[1] - head.p[1];
      const dz = wrist[2] - head.p[2];
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance < 20 || distance > 85) continue;
      if (this.gazeCosine(head, wrist) >= MENU_GAZE_COSINE) return side;
    }
    return null;
  }

  private lookedAtMenu(head: any): boolean {
    if (!head || !Array.isArray(head.p)) return false;
    const position = this.menu.getTransform().getWorldPosition();
    return this.gazeCosine(head, [position.x, position.y, position.z]) >= MENU_GAZE_COSINE;
  }

  private gazeCosine(head: any, target: number[]): number {
    const dx = target[0] - head.p[0];
    const dy = target[1] - head.p[1];
    const dz = target[2] - head.p[2];
    // Lens Studio cameras face along Transform.back, not Transform.forward.
    // The bridge accepts either axis, but a gaze test must use the visible one.
    const look = this.camera.getTransform().back;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const length = Math.sqrt(look.x * look.x + look.y * look.y + look.z * look.z);
    return length > 0 && distance > 0 ?
      (dx * look.x + dy * look.y + dz * look.z) / (distance * length) : -1;
  }

  private validWrist(hand: any): boolean {
    return !!hand && hand.tracked === true && Array.isArray(hand.wrist) && hand.wrist.length === 3;
  }

  private placeMenuByWrist(wrist: number[], head: any) {
    const right = this.camera.getTransform().right;
    const side = this.menuSide === "left" ? 1 : -1;
    const dx = head.p[0] - wrist[0];
    const dy = head.p[1] - wrist[1];
    const dz = head.p[2] - wrist[2];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    this.menu.getTransform().setWorldPosition(new vec3(
      wrist[0] + right.x * side * 10 + dx / distance * 4,
      wrist[1] + right.y * side * 10 + 5 + dy / distance * 4,
      wrist[2] + right.z * side * 10 + dz / distance * 4));
    this.menu.getTransform().setWorldRotation(this.camera.getTransform().getWorldRotation());
  }

  private parkMenu() {
    this.menu.getTransform().setWorldPosition(new vec3(0, -10000, 0));
  }

  private setPresentation(visible: boolean) {
    if (visible === this.presentationVisible) {
      return;
    }
    this.presentationVisible = visible;
    if (visible) {
      this.hideAt = 0;
      this.menuVisible = false;
      this.menuCloseAt = 0;
      this.menuNeedsLookAway = false;
      this.parkMenu();
      if (this.video !== null) this.video.enabled = true;
      return;
    }
    for (const side of ["left", "right"] as const) {
      this.tags[side].text.getSceneObject().enabled = false;
    }
    this.hideAt = getTime() + HIDE_DELAY_S;
  }

  private applyDelayedHide() {
    if (this.hideAt > 0 && getTime() >= this.hideAt) {
      this.hideAt = 0;
      this.menuVisible = false;
      this.parkMenu();
      if (this.video !== null) this.video.enabled = false;
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
    text.renderOrder = HUD_RENDER_ORDER + 1;
    text.depthTest = false;
    text.twoSided = true;
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

  private tagText(side: Side, guide: ArmGuide | null): string {
    const prefix = side === "left" ? "L" : "R";
    if (guide === null) return prefix + " · WAITING";
    const state = this.stateName(guide);
    const reason = guide.state && guide.state.indexOf(": at ") >= 0 ?
      guide.state.split(": at ")[1].toUpperCase() : "";
    if (reason) return prefix + " · " + state + "\n" + reason;
    const jaw = typeof guide.jaw === "number" ? (guide.jaw >= 30 ? "OPEN" : "CLOSED") : "GRIP --";
    const tip = typeof guide.tip === "number" ? Math.round(guide.tip) + " CM" : "--";
    return prefix + " · " + state + "\n" + jaw + " · " + tip;
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
