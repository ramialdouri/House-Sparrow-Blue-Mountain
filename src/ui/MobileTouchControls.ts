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
 * Refined 3-gesture touch & pointer input controller (Tap · Drag · Long Press)
 * with a floating flight-angle HUD ring:
 * 1. TAP (< 200ms, <= 15px movement) -> Crisp Hop / Jump (`onTapJump`).
 * 2. DRAG (> 16px movement before long-press threshold while grounded) -> Smooth Ground/Perch Move.
 * 3. LONG PRESS (>= 200ms hold, or hold-drag in flight) -> Launches into 30° upward flight;
 *    dragging up without releasing smoothly increases the flight angle from 30° up to 60° upwards,
 *    dragging down dives, dragging left/right banks & steers, and releasing initiates Glide.
 */
export class MobileTouchControls {
  private static readonly MAX_RADIUS = 48;
  private static readonly DEADZONE_PX = 6;
  private static readonly TAP_MAX_MOVE_PX = 15;
  private static readonly TAP_MAX_MS = 230;
  private static readonly LONG_PRESS_MS = 200;
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
  private upwardDragFlyTimer: number | null = null;
  private longPressActive = false;

  public readonly stick: MobileStickState = {
    active: false,
    x: 0,
    y: 0,
    magnitude: 0,
    longPress: false,
  };

  public onTapJump?: () => void;
  public onLongPressStart?: () => void;
  public onLongPressEnd?: () => void;
  public onTwoFingerTap?: () => boolean;
  public onFirstInteract?: () => void;
  public onMobileModeChange?: (isMobile: boolean) => void;
  /** Optional query so mid-air touches immediately engage flight control. */
  public isBirdFlying?: () => boolean;

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

  private startLongPressTimer(): void {
    this.clearLongPressTimer();
    if (this.isBirdFlying?.()) {
      // Already in the air: engage flight control almost immediately unless it's a quick tap
      this.longPressTimer = window.setTimeout(() => {
        this.triggerLongPress();
      }, 95);
      return;
    }
    this.longPressTimer = window.setTimeout(() => {
      if (!this.groundDragStarted) {
        this.triggerLongPress();
      }
    }, MobileTouchControls.LONG_PRESS_MS);
  }

  private clearLongPressTimer(): void {
    if (this.longPressTimer !== null) {
      window.clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }
    if (this.upwardDragFlyTimer !== null) {
      window.clearTimeout(this.upwardDragFlyTimer);
      this.upwardDragFlyTimer = null;
    }
  }

  private triggerLongPress(): void {
    this.clearLongPressTimer();
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
        if (this.isBirdFlying?.()) {
          // Mid-air drag immediately engages flight steering / climb / dive
          this.triggerLongPress();
        } else if (!this.groundDragStarted) {
          // Grounded/perched drag started before LONG_PRESS_MS -> enter Ground Drag mode
          this.groundDragStarted = true;
          this.clearLongPressTimer();
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
                this.onTapJump?.();
              }
            } else {
              this.onTapJump?.();
            }
          }
          this.secondaryId = null;
        } else if (t.identifier === this.primaryId) {
          this.clearLongPressTimer();
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
                  this.onTapJump?.();
                }
              } else {
                this.onTapJump?.();
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

    // Desktop pointer support: Tap = Hop, Drag = Move, Long Press = Fly (30° -> 60° on drag up), Release = Glide
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
          this.onTapJump?.();
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

    if (this.stick.longPress) {
      this.joystickRoot.classList.add('is-flying');
      if (this.stick.y < -0.22) {
        this.joystickRoot.classList.add('is-diving');
        this.joystickBadge.textContent = 'DIVE ↘';
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
    const nx = dist > 1e-4 ? (rawDx / dist) * clampedR : 0;
    const ny = dist > 1e-4 ? (rawDy / dist) * clampedR : 0;
    this.joystickNub.style.transform = `translate(calc(-50% + ${nx.toFixed(1)}px), calc(-50% + ${ny.toFixed(1)}px))`;
  }

  public resetAllTouches(): void {
    this.clearLongPressTimer();
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
