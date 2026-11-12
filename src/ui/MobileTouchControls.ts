export interface MobileStickState {
  active: boolean;
  /** Horizontal deflection in [-1, 1] (right = +1, left = -1). */
  x: number;
  /** Vertical deflection in [-1, 1] (screen up = +1, screen down = -1). */
  y: number;
  /** Deflection magnitude in [0, 1]. */
  magnitude: number;
  /** True while the user is actively long-pressing to fly. */
  longPress: boolean;
  /** Live pointer/touch screen coordinates in CSS pixels. */
  clientX: number;
  clientY: number;
  /** Touch-down origin screen coordinates in CSS pixels. */
  originX: number;
  originY: number;
}

export function detectInitialMobileDevice(): boolean {
  if (typeof window === 'undefined') return false;
  const uaMobile =
    typeof navigator !== 'undefined' &&
    /Android|iPhone|iPad|iPod|Mobile|CriOS|FxiOS/i.test(navigator.userAgent);
  if (uaMobile) return true;
  const coarse =
    typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
  const hasTouch = typeof navigator !== 'undefined' && navigator.maxTouchPoints > 0;
  if (coarse && hasTouch) return true;
  if (hasTouch && Math.min(window.innerWidth, window.innerHeight) <= 900) return true;
  return false;
}

/**
 * Refined 3-gesture touch & pointer input controller (Drag · Tap · Tap + Long Press)
 * with a floating flight-angle HUD ring:
 * 1. DRAG (> 15px movement while grounded/perched) -> Smooth Ground/Perch Move.
 * 2. TAP (<= 240ms, <= 15px movement) -> Crisp Hop (`onTapJump`), priming the hop-to-fly window.
 * 3. TAP + LONG PRESS (Tap to hop, then press & hold >= 85ms or drag in hop/flight) ->
 *    Zero-delay wing prime on touch-down, launching into 30° upward flight; dragging up without
 *    releasing smoothly increases climb angle from 30° up to 60°, dragging down dives,
 *    dragging left/right banks & steers, and releasing initiates a -10° paraglider glide.
 */
export class MobileTouchControls {
  private static readonly MAX_RADIUS = 48;
  private static readonly DEADZONE_PX = 6;
  private static readonly TAP_MAX_MOVE_PX = 15;
  private static readonly TAP_MAX_MS = 245;
  private static readonly HOP_FOLLOWUP_HOLD_MS = 85;
  private static readonly CHORD_WINDOW_MS = 85;

  private readonly joystickRoot: HTMLDivElement;
  private readonly joystickRing: HTMLDivElement;
  private readonly joystickNub: HTMLDivElement;
  private readonly joystickBadge: HTMLDivElement;

  private mobileActive: boolean;
  private primaryId: number | null = null;
  private originX = 0;
  private originY = 0;
  private curX = 0;
  private curY = 0;
  private primaryStartMs = 0;
  private primaryPeakDist = 0;
  private groundDragStarted = false;

  private secondaryId: number | null = null;
  private secondaryStartX = 0;
  private secondaryStartY = 0;
  private secondaryStartMs = 0;
  private secondaryPeakDist = 0;
  private chordCandidate = false;
  private chordFired = false;

  private longPressTimer: number | null = null;
  private longPressActive = false;
  private flyHoldPrimed = false;
  private pressEligibleForFly = false;

  public readonly stick: MobileStickState = {
    active: false,
    x: 0,
    y: 0,
    magnitude: 0,
    longPress: false,
    clientX: 0,
    clientY: 0,
    originX: 0,
    originY: 0,
  };

  public onTapJump?: (clientX: number, clientY: number) => void;
  public onFlyHoldPrime?: (primed: boolean) => void;
  public onLongPressStart?: () => void;
  public onLongPressEnd?: () => void;
  public onTwoFingerTap?: () => boolean;
  public onFirstInteract?: () => void;
  public onMobileModeChange?: (isMobile: boolean) => void;
  /** True when the bird is already in flying/gliding mode. */
  public isBirdFlying?: () => boolean;
  /** True when the bird just hopped via Tap and is ready to chain immediately into flight. */
  public isBirdHopReadyToFly?: () => boolean;

  constructor(parent: HTMLElement) {
    this.mobileActive = detectInitialMobileDevice();

    this.joystickRoot = document.createElement('div');
    this.joystickRoot.className = 'mobile-joystick';
    this.joystickRoot.setAttribute('aria-hidden', 'true');

    this.joystickRing = document.createElement('div');
    this.joystickRing.className = 'mobile-joystick-ring';

    this.joystickNub = document.createElement('div');
    this.joystickNub.className = 'mobile-joystick-nub';

    this.joystickBadge = document.createElement('div');
    this.joystickBadge.className = 'mobile-joystick-badge';
    this.joystickBadge.textContent = '30° ↗';

    this.joystickRoot.appendChild(this.joystickRing);
    this.joystickRoot.appendChild(this.joystickNub);
    this.joystickRoot.appendChild(this.joystickBadge);
    parent.appendChild(this.joystickRoot);

    this.bindTouchEvents();
  }

  public isMobile(): boolean {
    return this.mobileActive;
  }

  public isLongPressActive(): boolean {
    return this.longPressActive;
  }

  private setFlyHoldPrime(primed: boolean): void {
    if (this.flyHoldPrimed === primed) return;
    this.flyHoldPrimed = primed;
    this.onFlyHoldPrime?.(primed);
  }

  private startLongPressTimer(): void {
    this.clearLongPressTimer();
    const alreadyFlying = this.isBirdFlying?.() ?? false;
    const hopReady = this.isBirdHopReadyToFly?.() ?? false;

    // Grounded/perched without a preceding Tap-Hop: standalone Long Press does NOT fly.
    // The sparrow must hop first (Tap) before flying (Tap + Long Press).
    if (!alreadyFlying && !hopReady) {
      this.pressEligibleForFly = false;
      this.setFlyHoldPrime(false);
      return;
    }

    this.pressEligibleForFly = true;
    // Zero-delay anticipation: immediately prime wings open on touch-down so hop->fly has 0ms lag
    if (hopReady && !alreadyFlying) {
      this.setFlyHoldPrime(true);
    }

    this.longPressTimer = window.setTimeout(() => {
      if (this.primaryId !== null && this.pressEligibleForFly && !this.groundDragStarted) {
        this.triggerLongPress();
      }
    }, MobileTouchControls.HOP_FOLLOWUP_HOLD_MS);
  }

  private clearLongPressTimer(): void {
    if (this.longPressTimer !== null) {
      window.clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }
  }

  private triggerLongPress(): void {
    this.clearLongPressTimer();
    this.setFlyHoldPrime(false);
    if (!this.longPressActive && this.primaryId !== null) {
      this.longPressActive = true;
      this.groundDragStarted = false;
      this.stick.longPress = true;
      this.onLongPressStart?.();
      this.syncJoystickVisual();
    }
  }

  private endLongPress(): void {
    this.clearLongPressTimer();
    this.setFlyHoldPrime(false);
    if (this.longPressActive) {
      this.longPressActive = false;
      this.stick.longPress = false;
      this.onLongPressEnd?.();
      this.syncJoystickVisual();
    }
  }

  public setMobileOverride(enabled: boolean): void {
    if (this.mobileActive === enabled) return;
    this.mobileActive = enabled;
    if (!enabled) {
      this.resetAllTouches();
    }
    this.onMobileModeChange?.(enabled);
  }

  private activateMobileFromTouch(): void {
    if (!this.mobileActive) {
      this.mobileActive = true;
      this.onMobileModeChange?.(true);
    }
  }

  private isInteractiveUiTarget(target: EventTarget | null): boolean {
    if (!(target instanceof Element)) return false;
    return Boolean(
      target.closest('.journal-top-bar, .journal-card, .press-start-btn, .loading-screen:not(.is-loaded)')
    );
  }

  private handlePrimaryMove(clientX: number, clientY: number): void {
    this.curX = clientX;
    this.curY = clientY;
    const rawDx = this.curX - this.originX;
    const rawDy = this.curY - this.originY;
    const rawDist = Math.hypot(rawDx, rawDy);
    if (rawDist > this.primaryPeakDist) {
      this.primaryPeakDist = rawDist;
    }

    if (rawDist > MobileTouchControls.TAP_MAX_MOVE_PX) {
      this.chordCandidate = false;
      if (!this.longPressActive) {
        if (this.pressEligibleForFly || this.isBirdFlying?.()) {
          // Hop-ready or mid-air drag immediately engages flight steering / 30°->60° climb / dive
          this.updateStickFromDelta(rawDx, rawDy);
          this.triggerLongPress();
        } else if (!this.groundDragStarted) {
          // Grounded/perched drag without prior hop -> smooth Ground/Perch Move
          this.groundDragStarted = true;
          this.clearLongPressTimer();
          this.setFlyHoldPrime(false);
        }
      }
    }

    // Horizontal tether keeps left/right steering responsive; during Long-Press flight we preserve
    // a wide vertical anchor range so dragging up smoothly maps 30° -> 60° and returning lowers back to 30°.
    const maxHorizTether = MobileTouchControls.MAX_RADIUS * 1.45;
    if (Math.abs(rawDx) > maxHorizTether) {
      const pullX = Math.abs(rawDx) - maxHorizTether;
      this.originX += Math.sign(rawDx) * pullX;
    }
    const maxVertTether = this.longPressActive
      ? MobileTouchControls.MAX_RADIUS * 2.25
      : MobileTouchControls.MAX_RADIUS * 1.45;
    if (Math.abs(rawDy) > maxVertTether) {
      const pullY = Math.abs(rawDy) - maxVertTether;
      this.originY += Math.sign(rawDy) * pullY;
    }

    this.stick.clientX = this.curX;
    this.stick.clientY = this.curY;
    this.stick.originX = this.originX;
    this.stick.originY = this.originY;
    this.updateStickFromDelta(this.curX - this.originX, this.curY - this.originY);
    this.syncJoystickVisual();
  }

  private bindTouchEvents(): void {
    window.addEventListener(
      'touchstart',
      (e: TouchEvent) => {
        this.activateMobileFromTouch();
        this.onFirstInteract?.();

        if (this.isInteractiveUiTarget(e.target)) {
          return;
        }
        if (e.cancelable) {
          e.preventDefault();
        }

        const now = performance.now();
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (this.primaryId === null) {
            this.primaryId = t.identifier;
            this.originX = t.clientX;
            this.originY = t.clientY;
            this.curX = t.clientX;
            this.curY = t.clientY;
            this.primaryStartMs = now;
            this.primaryPeakDist = 0;
            this.groundDragStarted = false;
            this.chordCandidate = false;
            this.chordFired = false;
            this.stick.active = true;
            this.stick.longPress = false;
            this.stick.x = 0;
            this.stick.y = 0;
            this.stick.magnitude = 0;
            this.stick.clientX = t.clientX;
            this.stick.clientY = t.clientY;
            this.stick.originX = t.clientX;
            this.stick.originY = t.clientY;
            this.startLongPressTimer();
            this.syncJoystickVisual();
          } else if (this.secondaryId === null && t.identifier !== this.primaryId) {
            this.secondaryId = t.identifier;
            this.secondaryStartX = t.clientX;
            this.secondaryStartY = t.clientY;
            this.secondaryStartMs = now;
            this.secondaryPeakDist = 0;
            if (
              now - this.primaryStartMs <= MobileTouchControls.CHORD_WINDOW_MS &&
              this.primaryPeakDist <= MobileTouchControls.TAP_MAX_MOVE_PX
            ) {
              this.chordCandidate = true;
            }
          }
        }
      },
      { passive: false }
    );

    window.addEventListener(
      'touchmove',
      (e: TouchEvent) => {
        if (this.primaryId === null && this.secondaryId === null) return;
        if (e.cancelable) {
          e.preventDefault();
        }

        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (t.identifier === this.primaryId) {
            this.handlePrimaryMove(t.clientX, t.clientY);
          } else if (t.identifier === this.secondaryId) {
            const sDist = Math.hypot(
              t.clientX - this.secondaryStartX,
              t.clientY - this.secondaryStartY
            );
            if (sDist > this.secondaryPeakDist) {
              this.secondaryPeakDist = sDist;
            }
            if (sDist > MobileTouchControls.TAP_MAX_MOVE_PX + 4) {
              this.chordCandidate = false;
            }
          }
        }
      },
      { passive: false }
    );

    const handleTouchEnd = (e: TouchEvent, cancelled: boolean) => {
      const now = performance.now();
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];

        if (t.identifier === this.secondaryId) {
          const dur = now - this.secondaryStartMs;
          const isCleanTap =
            !cancelled &&
            dur <= MobileTouchControls.TAP_MAX_MS &&
            this.secondaryPeakDist <= MobileTouchControls.TAP_MAX_MOVE_PX + 4;

          if (isCleanTap) {
            if (this.chordCandidate && !this.chordFired) {
              this.chordFired = true;
              const consumedDrop = this.onTwoFingerTap?.() ?? false;
              if (!consumedDrop) {
                this.onTapJump?.(t.clientX, t.clientY);
              }
            } else {
              this.onTapJump?.(t.clientX, t.clientY);
            }
          }
          this.secondaryId = null;
        } else if (t.identifier === this.primaryId) {
          this.clearLongPressTimer();
          this.setFlyHoldPrime(false);
          this.pressEligibleForFly = false;
          const wasLong = this.longPressActive;
          if (wasLong) {
            this.endLongPress();
          } else {
            const dur = now - this.primaryStartMs;
            const isCleanTap =
              !cancelled &&
              !this.chordFired &&
              !this.groundDragStarted &&
              dur <= MobileTouchControls.TAP_MAX_MS &&
              this.primaryPeakDist <= MobileTouchControls.TAP_MAX_MOVE_PX;

            if (isCleanTap) {
              if (this.chordCandidate && this.secondaryId !== null) {
                this.chordFired = true;
                const consumedDrop = this.onTwoFingerTap?.() ?? false;
                if (!consumedDrop) {
                  this.onTapJump?.(t.clientX, t.clientY);
                }
              } else {
                this.onTapJump?.(t.clientX, t.clientY);
              }
            }
          }

          this.primaryId = null;
          this.groundDragStarted = false;
          this.stick.active = false;
          this.stick.longPress = false;
          this.stick.x = 0;
          this.stick.y = 0;
          this.stick.magnitude = 0;
          this.syncJoystickVisual();
        }
      }
    };

    window.addEventListener('touchend', (e) => handleTouchEnd(e, false), { passive: false });
    window.addEventListener('touchcancel', (e) => handleTouchEnd(e, true), { passive: false });

    // Desktop pointer support: Drag = Move, Tap = Hop, Tap + Long Press = Fly (30° -> 60° on drag up), Release = -10° Glide
    window.addEventListener('mousedown', (e: MouseEvent) => {
      if (e.button !== 0 || document.pointerLockElement !== null || this.primaryId !== null) return;
      if (this.isInteractiveUiTarget(e.target)) return;
      this.onFirstInteract?.();

      this.primaryId = -1;
      this.originX = e.clientX;
      this.originY = e.clientY;
      this.curX = e.clientX;
      this.curY = e.clientY;
      this.primaryStartMs = performance.now();
      this.primaryPeakDist = 0;
      this.groundDragStarted = false;
      this.chordCandidate = false;
      this.chordFired = false;
      this.stick.active = true;
      this.stick.longPress = false;
      this.stick.x = 0;
      this.stick.y = 0;
      this.stick.magnitude = 0;
      this.stick.clientX = e.clientX;
      this.stick.clientY = e.clientY;
      this.stick.originX = e.clientX;
      this.stick.originY = e.clientY;
      this.startLongPressTimer();
      this.syncJoystickVisual();
    });

    window.addEventListener('mousemove', (e: MouseEvent) => {
      if (this.primaryId !== -1) return;
      this.handlePrimaryMove(e.clientX, e.clientY);
    });

    window.addEventListener('mouseup', (e: MouseEvent) => {
      if (e.button !== 0 || this.primaryId !== -1) return;
      this.clearLongPressTimer();
      this.setFlyHoldPrime(false);
      this.pressEligibleForFly = false;
      const wasLong = this.longPressActive;
      if (wasLong) {
        this.endLongPress();
      } else {
        const dur = performance.now() - this.primaryStartMs;
        if (
          !this.groundDragStarted &&
          dur <= MobileTouchControls.TAP_MAX_MS &&
          this.primaryPeakDist <= MobileTouchControls.TAP_MAX_MOVE_PX
        ) {
          this.onTapJump?.(e.clientX, e.clientY);
        }
      }
      this.primaryId = null;
      this.groundDragStarted = false;
      this.stick.active = false;
      this.stick.longPress = false;
      this.stick.x = 0;
      this.stick.y = 0;
      this.stick.magnitude = 0;
      this.syncJoystickVisual();
    });

    window.addEventListener('blur', () => this.resetAllTouches());
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.resetAllTouches();
    });
  }

  private updateStickFromDelta(dx: number, dy: number): void {
    const dist = Math.hypot(dx, dy);
    const dead = MobileTouchControls.DEADZONE_PX;
    if (dist <= dead) {
      this.stick.x = 0;
      this.stick.y = 0;
      this.stick.magnitude = 0;
      return;
    }

    const maxR = MobileTouchControls.MAX_RADIUS;
    const span = Math.max(1, maxR - dead);

    // Independent axis normalization so steering horizontally never reduces vertical 30°->60° climb angle
    const absX = Math.abs(dx);
    const absY = Math.abs(dy);
    const normX = absX <= dead ? 0 : Math.sign(dx) * Math.min(1, (absX - dead) / span);
    const normY = absY <= dead ? 0 : Math.sign(-dy) * Math.min(1, (absY - dead) / span);

    this.stick.x = normX;
    this.stick.y = normY;
    this.stick.magnitude = Math.min(1, (dist - dead) / span);
  }

  private syncJoystickVisual(): void {
    if (!this.stick.active || this.primaryId === null) {
      this.joystickRoot.classList.remove('is-active', 'is-flying', 'is-diving');
      this.joystickNub.style.transform = 'translate(-50%, -50%)';
      return;
    }
    this.joystickRoot.classList.add('is-active');

    const isHelicopterDown =
      (this.stick.longPress || this.flyHoldPrimed) && this.stick.y < -0.16;

    if (this.stick.longPress || this.flyHoldPrimed) {
      this.joystickRoot.classList.add('is-flying');
      if (isHelicopterDown) {
        this.joystickRoot.classList.add('is-diving');
        this.joystickBadge.textContent = 'FLUTTER ↓';
      } else {
        this.joystickRoot.classList.remove('is-diving');
        const upDrag = Math.max(0, Math.min(1, this.stick.y));
        const deg = Math.round(30 + upDrag * 30);
        this.joystickBadge.textContent = `${deg}° ↗`;
      }
    } else {
      this.joystickRoot.classList.remove('is-flying', 'is-diving');
    }

    this.joystickRoot.style.transform = `translate3d(${this.originX.toFixed(1)}px, ${this.originY.toFixed(1)}px, 0)`;

    const rawDx = this.curX - this.originX;
    const rawDy = this.curY - this.originY;
    const dist = Math.hypot(rawDx, rawDy);
    const clampedR = Math.min(MobileTouchControls.MAX_RADIUS, dist);
    const nx = (dist > 1e-4 ? (rawDx / dist) * clampedR : 0) * (isHelicopterDown ? 0.14 : 1.0);
    const ny = dist > 1e-4 ? (rawDy / dist) * clampedR : 0;
    this.joystickNub.style.transform = `translate(calc(-50% + ${nx.toFixed(1)}px), calc(-50% + ${ny.toFixed(1)}px))`;
  }

  public resetAllTouches(): void {
    this.clearLongPressTimer();
    this.setFlyHoldPrime(false);
    this.pressEligibleForFly = false;
    if (this.longPressActive) {
      this.endLongPress();
    }
    this.primaryId = null;
    this.secondaryId = null;
    this.groundDragStarted = false;
    this.chordCandidate = false;
    this.chordFired = false;
    this.stick.active = false;
    this.stick.longPress = false;
    this.stick.x = 0;
    this.stick.y = 0;
    this.stick.magnitude = 0;
    this.syncJoystickVisual();
  }
}
