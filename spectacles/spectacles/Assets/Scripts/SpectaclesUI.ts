/** Spatial Spectacles UI: a movable video Frame, hand tags, and a palm-summoned menu. */

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
// The menu comes to a palm turned to the glasses and held there, not pinching:
// driving the arms is palm down, pinched. (It used to come to a look at a
// raised wrist, in a 30-degree cone -- which is where the eyes are while
// driving, so it kept opening and holding the arms.)
const MENU_DWELL_S = 0.6;
const MENU_HIDE_S = 1.0;
// It opens this far ahead, whatever the palm's distance, just below the line
// of sight and a little towards the other hand. At the wrist, as it was, a
// comfortably bent arm (palm 30-40 cm away) put 20 x 22 cm of menu nearer
// than 35 cm, past the edges of the glasses' ~30 x 35 degree view: the
// wearer held the palm out at 45-58 cm to open it, almost every time
// (2026-09-27 10:42 session).
const MENU_DISTANCE = 45;
const MENU_SCREEN_Y = 0.56;
const MENU_SHIFT = 0.05; // of the view's width
// Open this long whatever the hands do, then while the palm stays up, a
// fingertip is within reach of it, or a pinch holds: lower the palm and
// press with either hand.
const MENU_GRACE_S = 2.5;
const MENU_REACH_CM = 18;
const EMPTY_Y = -9.8; // below B601, shown only until the zone is photographed
// The hint card: this far ahead, catching up with the view by this share a frame.
const HINT_DISTANCE = 60;
const HINT_FOLLOW = 0.1;
const UI_OFF = "GLASSES UI IS OFF\nON THE WEB PAGE: ENTER SPECTACLES UI";

export class SpectaclesUI {
  private tags: { left: HandTag; right: HandTag };
  private view: Camera;
  private manualActive = false;
  private b601Active = false;
  private b601LiveActive = false;
  private b601Only = false;
  private so101Offline = false;
  private videoAvailable = true;
  private autoActive = false;
  private manualHighlight: boolean | null = null;
  private b601LiveHighlight: boolean | null = null;
  private autoHighlight: boolean | null = null;
  private manualLabel: Text;
  private b601LiveLabel: Text;
  private autoLabel: Text;
  private exitLabel: Text;
  private modeLabel: Text;
  private menu: SceneObject;
  private menuVisible = false;
  private menuSide: Side = "left";
  private palmSide: Side | null = null;
  private palmSince = 0;
  private lastPalm = 0;
  private lastMenuPinch = 0;
  private lastNear = 0;
  private menuOpenedAt = 0;
  private menuCloseAt = 0;
  private menuNeedsPalmDown = false;
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
  private hint: Text;

  constructor(private camera: SceneObject, private send: (command: Command) => void) {
    this.view = camera.getComponent("Component.Camera") as Camera;
    this.tags = { left: this.createTag("left"), right: this.createTag("right") };
    this.menu = this.createMenu();
    this.parkMenu();
    this.hint = this.createHint();
  }

  /**
   * Every frame, the Mac answering or not: the glasses must never show nothing.
   * With the glasses UI off -- as the web page starts, and after EXIT UI -- the
   * video goes and the menu with it, and Arm Status is off in the scene: the
   * Lens showed the video for a second and then nothing, which looks just like
   * a crash (2026-09-27: relaunched four times in a minute). A card in front
   * of the wearer now says why, and what to do.
   */
  public tick(linked: boolean, link: string) {
    this.applyDelayedHide();
    const words = !linked ? link : !this.presentationVisible && this.hideAt === 0 ? UI_OFF : "";
    const object = this.hint.getSceneObject();
    if (!words) {
      if (object.enabled) object.enabled = false;
      return;
    }
    this.setText(this.hint, words);
    const target = this.view.screenSpaceToWorldSpace(new vec2(0.5, 0.5), HINT_DISTANCE);
    const transform = object.getTransform();
    transform.setWorldPosition(object.enabled ?
      vec3.lerp(transform.getWorldPosition(), target, HINT_FOLLOW) : target);
    transform.setWorldRotation(this.camera.getTransform().getWorldRotation());
    if (!object.enabled) object.enabled = true;
  }

  public placeVideo(frame: SceneObject) {
    this.video = frame;
    this.place(frame, new vec2(0.5, 0.55), 78);
  }

  public updateReport(report: any) {
    this.applyDelayedHide();
    if (typeof report.standaloneB601 === "boolean" && report.standaloneB601 !== this.b601Only) {
      this.b601Only = report.standaloneB601;
      this.layoutB601Buttons();
    }
    if (typeof report.so101Offline === "boolean" && report.so101Offline !== this.so101Offline) {
      this.so101Offline = report.so101Offline;
      this.layoutB601Buttons();
    }
    if (typeof report.videoAvailable === "boolean" && report.videoAvailable !== this.videoAvailable) {
      this.videoAvailable = report.videoAvailable;
      if (this.video !== null) this.video.enabled = this.videoAvailable && this.presentationVisible;
    }
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
    this.b601Active = this.manualActive && report.manualTarget === "b601";
    this.b601LiveActive = this.manualActive && report.manualTarget === "b601_live";
    this.setText(this.b601LiveLabel, this.b601LiveActive ? "B601 HOLD" : "B601 LIVE");
    this.autoActive = report.auto === true;
    const arms = report.manual ? this.b601Active || this.b601LiveActive ?
      report.hands || {} : report.arms || report.hands || {} : {};
    for (const side of ["left", "right"] as const) {
      this.tags[side].guide = arms[side] && typeof arms[side] === "object" ? arms[side] as ArmGuide : null;
    }
    if (this.manualHighlight !== (this.manualActive && !this.b601Active && !this.b601LiveActive)) {
      this.highlight(this.manualLabel, this.manualActive && !this.b601Active && !this.b601LiveActive,
        new vec4(0.15, 0.8, 1, 1));
      this.manualHighlight = this.manualActive && !this.b601Active && !this.b601LiveActive;
    }
    if (this.b601LiveHighlight !== (this.b601Active || this.b601LiveActive)) {
      this.highlight(this.b601LiveLabel, this.b601Active || this.b601LiveActive,
        new vec4(1, 0.46, 0.28, 1));
      this.b601LiveHighlight = this.b601Active || this.b601LiveActive;
    }
    if (this.autoHighlight !== this.autoActive) {
      this.highlight(this.autoLabel, this.autoActive, new vec4(0.3, 1, 0.5, 1));
      this.autoHighlight = this.autoActive;
    }
    const error = typeof report.controlError === "string" ? report.controlError : "";
    const mode = error === "photograph the empty zone first" ? "CLEAR ZONE · TAP EMPTY" : error ||
      (report.busy === "empty" ? "CAPTURING EMPTY ZONE" :
       this.b601LiveActive ? "B601 LIVE · 8°/S · INDEX MOVE / MIDDLE TURN" :
       this.b601Active ? "B601-RS · HAND PREVIEW ONLY" :
       report.b601Parking ? "B601 · PARKING TO SLEEP POSE" :
       report.b601Holding ? "B601 HOLD · LIVE TO RESUME / NEUTRAL TO PARK" :
       this.b601Only ? "B601 READY · TAP B601" :
       this.manualActive ? "SO-101 · HAND CONTROL" : this.autoActive ? "AUTO · SORTING" :
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
        this.menuVisible ? "MENU OPEN\nPALM DOWN TO DRIVE" : this.tagText(side, tag.guide));
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

  private createHint(): Text {
    const object = global.scene.createSceneObject("Status Hint");
    const text = this.createText(object, 26, 5, 20);
    text.backgroundSettings.enabled = true;
    text.backgroundSettings.cornerRadius = 0.4;
    text.backgroundSettings.margins = Rect.create(1, 1, 0.6, 0.6);
    text.backgroundSettings.fill.color = new vec4(0.05, 0.07, 0.1, 0.88);
    object.enabled = false;
    return text;
  }

  private createMenu(): SceneObject {
    const menu = global.scene.createSceneObject("Palm Control Menu");
    this.place(menu, new vec2(0.5, 0.7), 55);
    const modeObject = global.scene.createSceneObject("Control Mode");
    modeObject.setParent(menu);
    modeObject.getTransform().setLocalPosition(new vec3(0, 9, 0));
    this.modeLabel = this.createText(modeObject, 16, 2, 14);
    this.modeLabel.text = "READY";
    // Targets a fingertip hits: 7 x 4 cm, 2 cm apart, the whole menu 16 x 18
    // cm (about 20 degrees at MENU_DISTANCE). Hand tracking is good to a
    // centimetre or two; the old 0.3-0.7 cm gaps took the neighbouring button.
    this.manualLabel = this.createButton(menu, "MANUAL", -4.5, 5, 7, 4, 20, () => {
      this.send({ command: "manual", enabled: !(this.manualActive && !this.b601Active) });
      this.queueMenuClose();
    });
    this.autoLabel = this.createButton(menu, "AUTO", 4.5, 5, 7, 4, 20, () => {
      this.send({ command: "auto", enabled: !this.autoActive });
      this.queueMenuClose();
    });
    const neutral = this.createButton(menu, "NEUTRAL", -4.5, -0.75, 7, 3.5, 18,
      () => {
        this.send({ command: "neutral" });
        this.queueMenuClose();
      });
    neutral.textFill.color = new vec4(1, 0.82, 0.35, 1);
    this.exitLabel = this.createButton(menu, "EXIT UI", 4.5, -0.75, 7, 3.5, 16, () => {
      this.send({ command: "presentation", enabled: false });
    });
    this.exitLabel.textFill.color = new vec4(0.72, 0.75, 0.82, 1);
    this.b601LiveLabel = this.createButton(menu, "B601 LIVE", 0, -5.6, 16, 3.2, 18, () => {
      this.send({ command: "b601_live", enabled: !this.b601LiveActive });
      this.queueMenuClose();
    });
    const empty = this.createButton(menu, "EMPTY ZONE", 0, EMPTY_Y, 16, 3.2, 16,
      () => this.send({ command: "empty" }));
    empty.textFill.color = new vec4(1, 0.82, 0.35, 1);
    this.emptyButton = empty.getSceneObject().getParent();
    this.positionEmptyButton();
    return menu;
  }

  private layoutB601Buttons() {
    // Keep the main Lens layout and Frame; only the fourth button changes
    // meaning because this B601-only bridge has no web controls to return to.
    this.setText(this.manualLabel, this.so101Offline ? "SO101 OFF" : "MANUAL");
    this.setText(this.autoLabel, this.so101Offline ? "AUTO OFF" : "AUTO");
    this.setText(this.exitLabel, this.b601Only ? "HOLD" : "EXIT UI");
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
    // Keep UIKit alive through trigger-up, then require the palm down before it reopens.
    this.menuCloseAt = getTime() + HIDE_DELAY_S;
  }

  private positionEmptyButton() {
    this.emptyButton.getTransform().setLocalPosition(new vec3(0, this.emptyNeeded ? EMPTY_Y : -10000, 0));
  }

  private updateMenu(message: any) {
    if (!this.presentationVisible) {
      this.palmSide = null;
      return;
    }
    const now = getTime();
    const palm = this.palmUp(message);
    if (this.menuNeedsPalmDown) {
      if (palm === null) {
        this.menuNeedsPalmDown = false;
        this.palmSide = null;
        this.palmSince = now;
      }
      return;
    }
    if (palm !== this.palmSide) {
      this.palmSide = palm;
      this.palmSince = now;
    }
    if (!this.menuVisible && palm !== null && now - this.palmSince >= MENU_DWELL_S) {
      this.menuSide = palm;
      this.positionEmptyButton();
      this.placeMenu();
      this.menuVisible = true;
      this.menuOpenedAt = now;
      this.lastPalm = now;
    }
    if (!this.menuVisible) return;
    if (palm === this.menuSide) this.lastPalm = now;
    if (this.fingertipNearMenu(message)) this.lastNear = now;
    const pinching = !!((message.left && message.left.pinch) || (message.right && message.right.pinch));
    if (pinching) this.lastMenuPinch = now;
    if (this.menuCloseAt > 0 && now >= this.menuCloseAt && !pinching) {
      this.menuCloseAt = 0;
      this.closeMenu(now);
      this.menuNeedsPalmDown = true;
      return;
    }
    const inUse = Math.max(this.lastPalm, this.lastMenuPinch, this.lastNear, this.menuOpenedAt + MENU_GRACE_S);
    if (!pinching && now - inUse > MENU_HIDE_S) {
      this.closeMenu(now);
    }
  }

  /** Either index fingertip within reach of the open menu: about to press it. */
  private fingertipNearMenu(message: any): boolean {
    const centre = this.menu.getTransform().getWorldPosition();
    for (const side of ["left", "right"] as const) {
      const hand = message && message[side];
      if (!hand || hand.tracked !== true || !Array.isArray(hand.index) || hand.index.length !== 3) continue;
      const tip = new vec3(hand.index[0], hand.index[1], hand.index[2]);
      if (tip.distance(centre) <= MENU_REACH_CM) return true;
    }
    return false;
  }

  private closeMenu(now: number) {
    this.menuVisible = false;
    // The finger that pressed may still be pinched: it must let go before it drives.
    this.uiNeedsRelease = true;
    this.uiReleaseAfter = now + 0.35;
    this.parkMenu();
  }

  /** The hand holding its palm to the glasses, raised in front and not pinching, if any. */
  private palmUp(message: any): Side | null {
    const head = message && message.head;
    if (!head || !Array.isArray(head.p)) return null;
    for (const side of ["left", "right"] as const) {
      const hand = message[side];
      if (!this.validWrist(hand) || hand.palm !== true || hand.pinch === true) continue;
      const dx = hand.wrist[0] - head.p[0];
      const dy = hand.wrist[1] - head.p[1];
      const dz = hand.wrist[2] - head.p[2];
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance >= 20 && distance <= 75) return side;
    }
    return null;
  }

  private validWrist(hand: any): boolean {
    return !!hand && hand.tracked === true && Array.isArray(hand.wrist) && hand.wrist.length === 3;
  }

  /** In view at MENU_DISTANCE, a little towards the other hand, and there it stays. */
  private placeMenu() {
    const towardsOtherHand = this.menuSide === "left" ? 1 : -1;
    this.menu.getTransform().setWorldPosition(this.view.screenSpaceToWorldSpace(
      new vec2(0.5 + towardsOtherHand * MENU_SHIFT, MENU_SCREEN_Y), MENU_DISTANCE));
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
      this.menuNeedsPalmDown = false;
      this.parkMenu();
      if (this.video !== null) this.video.enabled = this.videoAvailable;
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
    if (this.b601LiveActive) return "B601 · " + state + "\n" + jaw + " · 8°/S";
    if (this.b601Active) return "B601 · " + state + "\nPREVIEW ONLY";
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
