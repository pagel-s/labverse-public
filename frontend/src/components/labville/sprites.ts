import Phaser from "phaser";

import type { Person } from "../../api/client";
import { initials } from "../../lib/lab";

const ROLE_COLORS: Record<string, number> = {
  pi: 0x8c5a18,
  postdoc: 0x1f7a6f,
  grad_student: 0x2f7ebc,
  technician: 0x6b7a8c,
  research_assistant: 0x5b8c6b,
};

/** Map role to character sprite index (column in characters.png). */
const ROLE_SPRITE_INDEX: Record<string, number> = {
  pi: 0,
  postdoc: 1,
  grad_student: 2,
  technician: 4,
  research_assistant: 3,
};

const CHAR_W = 20;
const CHAR_H = 28;

const BUBBLE_STYLE: Phaser.Types.GameObjects.Text.TextStyle = {
  fontSize: "9px",
  fontFamily: '"IBM Plex Mono", monospace',
  color: "#2b2b2b",
  wordWrap: { width: 100 },
};

const NAME_STYLE: Phaser.Types.GameObjects.Text.TextStyle = {
  fontSize: "8px",
  fontFamily: '"Avenir Next", "Segoe UI", sans-serif',
  color: "#e8ecf0",
  fontStyle: "bold",
};

export class AgentSprite extends Phaser.GameObjects.Container {
  personId: string;
  private disc: Phaser.GameObjects.Ellipse;
  private charSprite: Phaser.GameObjects.Image | null = null;
  private label: Phaser.GameObjects.Text;
  private nameTag: Phaser.GameObjects.Text;
  private nameTagBg: Phaser.GameObjects.Graphics;
  private bubble: Phaser.GameObjects.Container | null = null;
  private highlight: Phaser.GameObjects.Ellipse;
  private shadow: Phaser.GameObjects.Ellipse;
  private currentActivity: string | null = null;
  private bubbleTimer: Phaser.Time.TimerEvent | null = null;
  private bodyOffset = -4;

  constructor(scene: Phaser.Scene, person: Person, x: number, y: number) {
    super(scene, x, y);
    this.personId = person.id;

    const color = ROLE_COLORS[person.role] ?? 0x6b7a8c;

    // Shadow under character
    this.shadow = new Phaser.GameObjects.Ellipse(scene, 0, 12, 24, 10, 0x000000, 0.18);
    this.add(this.shadow);

    // Selection highlight ring (hidden by default)
    this.highlight = new Phaser.GameObjects.Ellipse(scene, 0, 0, 36, 36, 0x1f7a6f, 0.25);
    this.highlight.setVisible(false);
    this.add(this.highlight);

    // Try to create character sprite from the spritesheet
    const spriteIdx = ROLE_SPRITE_INDEX[person.role] ?? 0;
    if (scene.textures.exists("characters")) {
      const texKey = `char-${person.id}`;
      if (!scene.textures.exists(texKey)) {
        const canvas = document.createElement("canvas");
        canvas.width = CHAR_W;
        canvas.height = CHAR_H;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const srcImage = scene.textures.get("characters").getSourceImage() as HTMLImageElement;
          ctx.drawImage(srcImage, spriteIdx * CHAR_W, 0, CHAR_W, CHAR_H, 0, 0, CHAR_W, CHAR_H);
          scene.textures.addCanvas(texKey, canvas);
        }
      }
      if (scene.textures.exists(texKey)) {
        this.charSprite = new Phaser.GameObjects.Image(scene, 0, -4, texKey);
        this.charSprite.setScale(1.6);
        this.add(this.charSprite);
      }
    }
    this.bodyOffset = this.charSprite ? -4 : 0;

    // Fallback colored disc with initials (if no sprite)
    this.disc = new Phaser.GameObjects.Ellipse(scene, 0, 0, 24, 24, color, 1);
    this.disc.setStrokeStyle(2, 0xffffff, 0.9);
    if (this.charSprite) {
      this.disc.setVisible(false);
    }
    this.add(this.disc);

    // Initials label (shown only if no sprite)
    this.label = new Phaser.GameObjects.Text(scene, 0, 0, initials(person.name), {
      fontSize: "10px",
      fontFamily: '"IBM Plex Mono", monospace',
      color: "#ffffff",
      fontStyle: "bold",
    });
    this.label.setOrigin(0.5);
    if (this.charSprite) this.label.setVisible(false);
    this.add(this.label);

    // Name tag with subtle background for readability
    const firstName = person.name.split(" ")[0] ?? "";
    this.nameTagBg = new Phaser.GameObjects.Graphics(scene);
    this.add(this.nameTagBg);

    this.nameTag = new Phaser.GameObjects.Text(scene, 0, 20, firstName, NAME_STYLE);
    this.nameTag.setOrigin(0.5, 0);
    this.add(this.nameTag);

    // Draw name background
    this.drawNameBg();

    this.startIdleMotion();

    // Make interactive
    this.setSize(32, 40);
    this.setInteractive({ useHandCursor: true });

    scene.add.existing(this as Phaser.GameObjects.GameObject);
  }

  private drawNameBg(): void {
    this.nameTagBg.clear();
    const tw = this.nameTag.width + 8;
    const th = this.nameTag.height + 2;
    this.nameTagBg.fillStyle(0x1a1e24, 0.6);
    this.nameTagBg.fillRoundedRect(-tw / 2, 19, tw, th, 3);
  }

  private startIdleMotion(): void {
    const bodyTargets: Array<
      Phaser.GameObjects.Image | Phaser.GameObjects.Ellipse | Phaser.GameObjects.Text
    > = [this.charSprite ?? this.disc];
    if (!this.charSprite) {
      bodyTargets.push(this.label);
    }

    this.scene.tweens.add({
      targets: bodyTargets,
      y: this.bodyOffset - 1.5,
      duration: 1500 + Math.random() * 450,
      yoyo: true,
      repeat: -1,
      ease: "Sine.easeInOut",
    });

    this.scene.tweens.add({
      targets: [this.nameTag, this.nameTagBg],
      y: "+=0.8",
      duration: 1500 + Math.random() * 450,
      yoyo: true,
      repeat: -1,
      ease: "Sine.easeInOut",
    });

    this.scene.tweens.add({
      targets: this.shadow,
      scaleX: 0.92,
      scaleY: 0.88,
      alpha: 0.14,
      duration: 1500 + Math.random() * 450,
      yoyo: true,
      repeat: -1,
      ease: "Sine.easeInOut",
    });
  }

  /** Tween-move to a new pixel position. */
  tweenTo(px: number, py: number, duration?: number): void {
    this.scene.tweens.killTweensOf(this);
    this.scene.tweens.killTweensOf(this.shadow);
    if (Math.abs(this.x - px) < 2 && Math.abs(this.y - py) < 2) return;
    const travel = Phaser.Math.Distance.Between(this.x, this.y, px, py);
    const moveDuration = duration ?? Phaser.Math.Clamp(travel * 6, 240, 760);

    if (this.charSprite) {
      this.charSprite.setFlipX(px < this.x);
    }

    this.scene.tweens.add({
      targets: this.shadow,
      scaleX: 1.08,
      scaleY: 0.8,
      duration: moveDuration * 0.5,
      yoyo: true,
      ease: "Quad.easeOut",
    });

    this.scene.tweens.add({
      targets: this,
      x: px,
      y: py,
      duration: moveDuration,
      ease: "Cubic.easeInOut",
    });
  }

  tweenAlong(points: Array<{ x: number; y: number }>): void {
    const steps = points.filter((point, index) => {
      if (index === 0) {
        return Phaser.Math.Distance.Between(this.x, this.y, point.x, point.y) > 2;
      }
      const previous = points[index - 1];
      return Phaser.Math.Distance.Between(previous.x, previous.y, point.x, point.y) > 2;
    });
    if (steps.length === 0) {
      return;
    }

    this.scene.tweens.killTweensOf(this);
    this.scene.tweens.killTweensOf(this.shadow);

    let cursorX = this.x;
    let cursorY = this.y;
    const tweenChain = steps.map((point) => {
      const travel = Phaser.Math.Distance.Between(cursorX, cursorY, point.x, point.y);
      const duration = Phaser.Math.Clamp(travel * 6.4, 180, 620);
      const facingLeft = point.x < cursorX;
      cursorX = point.x;
      cursorY = point.y;
      return {
        x: point.x,
        y: point.y,
        duration,
        ease: "Cubic.easeInOut",
        onStart: () => {
          if (this.charSprite) {
            this.charSprite.setFlipX(facingLeft);
          }
          this.scene.tweens.add({
            targets: this.shadow,
            scaleX: 1.08,
            scaleY: 0.8,
            duration: duration * 0.5,
            yoyo: true,
            ease: "Quad.easeOut",
          });
        },
      };
    });

    this.scene.tweens.chain({
      targets: this,
      tweens: tweenChain,
    });
  }

  /** Show/hide selection highlight. */
  setHighlighted(on: boolean): void {
    this.highlight.setVisible(on);
    if (on) {
      this.scene.tweens.add({
        targets: this.highlight,
        scaleX: 1.3,
        scaleY: 1.3,
        alpha: 0.12,
        duration: 800,
        yoyo: true,
        repeat: -1,
        ease: "Sine.easeInOut",
      });
    } else {
      this.scene.tweens.killTweensOf(this.highlight);
      this.highlight.setScale(1);
      this.highlight.setAlpha(0.25);
    }
  }

  /** Show activity speech bubble above agent. */
  showActivity(text: string | null): void {
    if (text === this.currentActivity) return;
    this.currentActivity = text;

    if (this.bubble) {
      this.bubble.destroy();
      this.bubble = null;
    }
    if (this.bubbleTimer) {
      this.bubbleTimer.destroy();
      this.bubbleTimer = null;
    }
    if (!text) return;

    const trimmed = text.length > 40 ? text.slice(0, 38) + "..." : text;
    const bubbleText = new Phaser.GameObjects.Text(this.scene, 6, 4, trimmed, BUBBLE_STYLE);
    const tw = Math.min(bubbleText.width + 12, 112);
    const th = bubbleText.height + 8;

    const bg = new Phaser.GameObjects.Graphics(this.scene);
    bg.fillStyle(0xffffff, 0.94);
    bg.lineStyle(1, 0xcccccc, 0.6);
    bg.fillRoundedRect(0, 0, tw, th, 5);
    bg.strokeRoundedRect(0, 0, tw, th, 5);
    // Triangle pointer
    bg.fillStyle(0xffffff, 0.94);
    bg.fillTriangle(tw / 2 - 4, th, tw / 2, th + 6, tw / 2 + 4, th);

    this.bubble = new Phaser.GameObjects.Container(this.scene, -tw / 2, -th - 26);
    this.bubble.add(bg);
    this.bubble.add(bubbleText);
    this.add(this.bubble);

    this.bubbleTimer = this.scene.time.delayedCall(5000, () => {
      if (this.bubble) {
        this.scene.tweens.add({
          targets: this.bubble,
          alpha: 0,
          duration: 500,
          onComplete: () => {
            this.bubble?.destroy();
            this.bubble = null;
          },
        });
      }
    });
  }

  /** Update from new person data. */
  updatePerson(person: Person): void {
    const color = ROLE_COLORS[person.role] ?? 0x6b7a8c;
    this.disc.setFillStyle(color, 1);
    this.label.setText(initials(person.name));
    this.nameTag.setText(person.name.split(" ")[0] ?? "");
    this.showActivity(person.current_activity);
  }
}
