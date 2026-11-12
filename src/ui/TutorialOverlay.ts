export type TutorialPromptKey =
  | 'WASD'
  | 'Q'
  | 'CLICK'
  | 'SPACE'
  | 'MOBILE_DOOR'
  | 'MOBILE_WINDOW'
  | null;

interface SingleKeySpec {
  label: string;
  codes: string[];
  wide?: boolean;
  medium?: boolean;
}

interface MobileChipSpec {
  gesture: string;
  hint: string;
  codes: string[];
}

const WASD_KEYS: ReadonlyArray<{ label: string; slotClass: string; codes: string[] }> = [
  { label: 'W', slotClass: 'wasd-w', codes: ['KeyW', 'ArrowUp'] },
  { label: 'A', slotClass: 'wasd-a', codes: ['KeyA', 'ArrowLeft'] },
  { label: 'S', slotClass: 'wasd-s', codes: ['KeyS', 'ArrowDown'] },
  { label: 'D', slotClass: 'wasd-d', codes: ['KeyD', 'ArrowRight'] },
];

const SINGLE_PROMPT_SPECS: Record<'Q' | 'CLICK' | 'SPACE', SingleKeySpec> = {
  Q: {
    label: 'Q',
    codes: ['KeyQ', 'KeyC', 'KeyR'],
  },
  CLICK: {
    label: 'CLICK',
    codes: ['MouseLeft'],
    medium: true,
  },
  SPACE: {
    label: 'SPACE',
    codes: ['Space'],
    wide: true,
  },
};

function resolveMobileChips(prompt: TutorialPromptKey): MobileChipSpec[] | null {
  switch (prompt) {
    case 'WASD':
      return [
        { gesture: 'DRAG', hint: 'Move', codes: ['TouchMove', 'KeyW', 'KeyA', 'KeyS', 'KeyD'] },
        { gesture: 'TAP', hint: 'Hop', codes: ['TouchTap', 'Space'] },
        { gesture: 'TAP + LONG PRESS', hint: 'Fly', codes: ['TouchTapHold'] },
      ];
    case 'MOBILE_DOOR':
      return [
        { gesture: 'DRAG', hint: 'To Cage Door', codes: ['TouchMove'] },
        { gesture: 'TAP + LONG PRESS', hint: 'Fly', codes: ['TouchTapHold'] },
      ];
    case 'SPACE':
      return [
        { gesture: 'TAP + LONG PRESS', hint: 'Fly (30°)', codes: ['TouchTapHold', 'Space'] },
        { gesture: 'DRAG UP', hint: 'Climb (60°)', codes: ['TouchMove', 'KeyW', 'ArrowUp'] },
        { gesture: 'RELEASE', hint: 'Glide (-10°)', codes: ['TouchTap'] },
      ];
    case 'MOBILE_WINDOW':
    case 'CLICK':
      return [
        { gesture: 'TAP + LONG PRESS', hint: 'Fly', codes: ['TouchTapHold', 'Space'] },
        { gesture: 'DRAG', hint: 'Steer / Climb', codes: ['TouchMove'] },
      ];
    case 'Q':
    case null:
    default:
      return null;
  }
}

/**
 * Contextual indie tutorial overlay:
 * - Level start onboarding shows all 3 core controls: DRAG = Move, TAP = Hop, TAP + LONG PRESS = Fly
 * - Desktop: physical 4-square inverted-T cluster for WASD + gesture pills so both keyboard and mouse/touch players see Tap + Long Press = Fly
 * - Mobile: clean tactile touch gesture pills (DRAG Move · TAP Hop · TAP + LONG PRESS Fly)
 * Built with safe DOM APIs (createElement, textContent, replaceChildren; no innerHTML).
 */
export class TutorialOverlay {
  private readonly root: HTMLDivElement;
  private readonly keyRow: HTMLDivElement;

  private requestedPrompt: TutorialPromptKey = null;
  private renderedSignature = '';
  private finished = false;
  private mobileMode = false;
  private activeKeyEls: Array<{ el: HTMLElement; codes: string[] }> = [];
  private holdTimer: number | null = null;
  private recentTapAtMs = -10000;
  private pressStartMs = 0;

  constructor(parent: HTMLElement) {
    this.root = document.createElement('div');
    this.root.className = 'tutorial-overlay is-idle';
    this.root.setAttribute('aria-hidden', 'true');

    this.keyRow = document.createElement('div');
    this.keyRow.className = 'tutorial-keys';

    this.root.appendChild(this.keyRow);
    parent.appendChild(this.root);

    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (e.code === 'Space') {
        const now = performance.now();
        const hadRecentTap = now - this.recentTapAtMs <= 800;
        this.pressStartMs = now;
        if (hadRecentTap) {
          this.startHoldFeedbackTimer();
        }
      }
      this.handleKeyState(e.code, true);
    });
    window.addEventListener('keyup', (e) => {
      if (e.code === 'Space') {
        const now = performance.now();
        if (now - this.pressStartMs <= 260) {
          this.recentTapAtMs = now;
        }
        this.clearHoldFeedbackTimer();
        this.handleKeyState('TouchTapHold', false);
      }
      this.handleKeyState(e.code, false);
    });
    window.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        const now = performance.now();
        const hadRecentTap = now - this.recentTapAtMs <= 800;
        this.pressStartMs = now;
        this.handleKeyState('MouseLeft', true);
        this.handleKeyState('TouchTap', true);
        if (hadRecentTap) {
          this.startHoldFeedbackTimer();
        }
      }
    });
    window.addEventListener('mousemove', (e) => {
      if ((e.buttons & 1) === 1) {
        this.handleKeyState('TouchMove', true);
      }
    });
    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) {
        const now = performance.now();
        if (now - this.pressStartMs <= 260) {
          this.recentTapAtMs = now;
        }
        this.clearHoldFeedbackTimer();
        this.handleKeyState('MouseLeft', false);
        this.handleKeyState('TouchTap', false);
        this.handleKeyState('TouchMove', false);
        this.handleKeyState('TouchTapHold', false);
      }
    });
    window.addEventListener(
      'touchstart',
      () => {
        const now = performance.now();
        const hadRecentTap = now - this.recentTapAtMs <= 800;
        this.pressStartMs = now;
        this.handleKeyState('TouchTap', true);
        if (hadRecentTap) {
          this.startHoldFeedbackTimer();
        }
      },
      { passive: true }
    );
    window.addEventListener(
      'touchmove',
      () => {
        this.handleKeyState('TouchMove', true);
      },
      { passive: true }
    );
    const clearTouch = () => {
      const now = performance.now();
      if (now - this.pressStartMs <= 260) {
        this.recentTapAtMs = now;
      }
      this.clearHoldFeedbackTimer();
      this.handleKeyState('TouchTap', false);
      this.handleKeyState('TouchMove', false);
      this.handleKeyState('TouchTapHold', false);
    };
    window.addEventListener('touchend', clearTouch, { passive: true });
    window.addEventListener('touchcancel', clearTouch, { passive: true });

    const clearPressed = () => {
      this.clearHoldFeedbackTimer();
      for (let i = 0; i < this.activeKeyEls.length; i++) {
        this.activeKeyEls[i].el.classList.remove('is-pressed');
      }
    };
    window.addEventListener('blur', clearPressed);
    document.addEventListener('visibilitychange', clearPressed);
  }

  private startHoldFeedbackTimer(): void {
    this.clearHoldFeedbackTimer();
    this.holdTimer = window.setTimeout(() => {
      this.handleKeyState('TouchTapHold', true);
    }, 85);
  }

  private clearHoldFeedbackTimer(): void {
    if (this.holdTimer !== null) {
      window.clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
  }

  private handleKeyState(code: string, down: boolean): void {
    for (let i = 0; i < this.activeKeyEls.length; i++) {
      const entry = this.activeKeyEls[i];
      if (entry.codes.includes(code)) {
        entry.el.classList.toggle('is-pressed', down);
      }
    }
  }

  public setMobileMode(isMobile: boolean): void {
    if (this.mobileMode === isMobile) return;
    this.mobileMode = isMobile;
    this.root.classList.toggle('is-mobile', isMobile);
    if (!this.finished) {
      this.renderCurrentPrompt();
    }
  }

  public setPrompt(prompt: TutorialPromptKey): void {
    if (this.finished) return;
    this.requestedPrompt = prompt;
    this.renderCurrentPrompt();
  }

  private buildMobilePillRow(chips: MobileChipSpec[]): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'mobile-tutorial-row';
    for (let i = 0; i < chips.length; i++) {
      const chip = chips[i];
      const pill = document.createElement('div');
      pill.className = 'mobile-touch-pill';

      const gestureEl = document.createElement('span');
      gestureEl.className = 'mobile-touch-gesture';
      gestureEl.textContent = chip.gesture;

      const hintEl = document.createElement('span');
      hintEl.className = 'mobile-touch-hint';
      hintEl.textContent = chip.hint;

      pill.appendChild(gestureEl);
      pill.appendChild(hintEl);
      this.activeKeyEls.push({ el: pill, codes: chip.codes });
      wrap.appendChild(pill);
    }
    return wrap;
  }

  private renderCurrentPrompt(): void {
    if (this.finished) return;
    const prompt = this.requestedPrompt;
    const sig = `${this.mobileMode ? 'M' : 'D'}:${prompt ?? 'NONE'}`;
    if (sig === this.renderedSignature) return;
    this.renderedSignature = sig;

    if (this.mobileMode) {
      const chips = resolveMobileChips(prompt);
      if (!chips || chips.length === 0) {
        this.activeKeyEls = [];
        this.root.classList.add('is-idle');
        this.root.setAttribute('aria-hidden', 'true');
        return;
      }

      this.keyRow.replaceChildren();
      this.activeKeyEls = [];
      this.keyRow.appendChild(this.buildMobilePillRow(chips));
    } else {
      if (prompt === null || prompt === 'MOBILE_DOOR' || prompt === 'MOBILE_WINDOW') {
        this.activeKeyEls = [];
        this.root.classList.add('is-idle');
        this.root.setAttribute('aria-hidden', 'true');
        return;
      }

      this.keyRow.replaceChildren();
      this.activeKeyEls = [];

      if (prompt === 'WASD') {
        const stack = document.createElement('div');
        stack.className = 'tutorial-onboarding-stack';

        const cluster = document.createElement('div');
        cluster.className = 'wasd-cluster';
        for (let i = 0; i < WASD_KEYS.length; i++) {
          const keySpec = WASD_KEYS[i];
          const kbd = document.createElement('kbd');
          kbd.className = `indie-key is-square ${keySpec.slotClass}`;
          kbd.textContent = keySpec.label;
          this.activeKeyEls.push({ el: kbd, codes: keySpec.codes });
          cluster.appendChild(kbd);
        }
        stack.appendChild(cluster);

        const chips = resolveMobileChips('WASD');
        if (chips) {
          stack.appendChild(this.buildMobilePillRow(chips));
        }
        this.keyRow.appendChild(stack);
      } else if (prompt === 'SPACE') {
        const stack = document.createElement('div');
        stack.className = 'tutorial-onboarding-stack';

        const spec = SINGLE_PROMPT_SPECS.SPACE;
        const kbd = document.createElement('kbd');
        kbd.className = 'indie-key is-wide';
        kbd.textContent = spec.label;
        this.activeKeyEls.push({ el: kbd, codes: spec.codes });
        stack.appendChild(kbd);

        const chips = resolveMobileChips('SPACE');
        if (chips) {
          stack.appendChild(this.buildMobilePillRow(chips));
        }
        this.keyRow.appendChild(stack);
      } else {
        const spec = SINGLE_PROMPT_SPECS[prompt];
        const kbd = document.createElement('kbd');
        kbd.className = spec.wide
          ? 'indie-key is-wide'
          : spec.medium
            ? 'indie-key is-medium'
            : 'indie-key';
        kbd.textContent = spec.label;
        this.activeKeyEls.push({ el: kbd, codes: spec.codes });
        this.keyRow.appendChild(kbd);
      }
    }

    this.root.classList.remove('is-idle', 'is-complete');
    this.root.setAttribute('aria-hidden', 'false');
    this.keyRow.classList.remove('step-pop');
    void this.keyRow.offsetWidth;
    this.keyRow.classList.add('step-pop');
  }

  public complete(): void {
    if (this.finished) return;
    this.finished = true;
    this.requestedPrompt = null;
    this.renderedSignature = 'DONE';
    this.root.classList.remove('is-idle');
    this.root.classList.add('is-complete');
    this.root.setAttribute('aria-hidden', 'true');
  }
}
