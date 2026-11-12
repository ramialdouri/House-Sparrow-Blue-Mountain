import * as THREE from 'three/webgpu';
import type * as CANNON from 'cannon-es';
import { SparrowMesh, type DizzyPoseState } from './SparrowMesh.ts';
import { createPerchQueryResult, type PerchSystem, type PerchSegment } from '../world/PerchSystem.ts';
import {
  createGroundRaycastHit,
  type PhysicsWorld,
  type InteractiveProp,
  type StaticCameraHit,
} from '../physics/PhysicsWorld.ts';
import type { AcousticSoundtrack, SurfaceSoundType } from '../audio/AcousticSoundtrack.ts';
import type { WorldFX } from '../rendering/WorldFX.ts';
import { PALETTE } from '../rendering/Palette.ts';
import { isInsideAtelier } from '../world/AtelierBounds.ts';
import type { MobileTouchControls, MobileStickState } from '../ui/MobileTouchControls.ts';

export type BirdMode = 'grounded' | 'perched' | 'flying';

export interface AttentionTarget {
  position: THREE.Vector3;
  priority: number;
}

export interface CageKeepInBounds {
  center: THREE.Vector3;
  radius: number;
  floorY: number;
  ceilingY: number;
  locked: boolean;
}

export interface ClosedWindowBarrier {
  xMax: number;
  yMin: number;
  yMax: number;
  zMin: number;
  zMax: number;
  closed: boolean;
}

const GROUND_FOOT_CLEARANCE = 0.384;
const PERCH_FOOT_CLEARANCE = 0.352;
const LEDGE_PROBE_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [0.65, 0],
  [-0.65, 0],
  [0, 0.65],
  [0, -0.65],
];
const GROUND_WALL_ANGLES: readonly number[] = [0, -0.52, 0.52, -1.12, 1.12];
const FLY_WALL_WHISKERS: readonly number[] = [0, -0.62, 0.62, -1.22, 1.22];

// Conservatory Atelier watertight interior & East window-aperture bounds
const ATELIER_INNER_X_MIN = -85.36;
const ATELIER_INNER_X_MAX = -40.12;
const ATELIER_INNER_Z_MIN = 26.62;
const ATELIER_INNER_Z_MAX = 77.38;
const ATELIER_INNER_Y_MIN = 26.2 + GROUND_FOOT_CLEARANCE;
const ATELIER_INNER_Y_MAX = 50.92;
const ATELIER_APERTURE_Z_MIN = 43.95;
const ATELIER_APERTURE_Z_MAX = 60.05;
const ATELIER_APERTURE_Y_MIN = 31.38;
const ATELIER_APERTURE_Y_MAX = 47.25;
const ATELIER_OUTER_X_EAST = -37.45;

const _perchQueryScratch = createPerchQueryResult();

const _down = new THREE.Vector3(0, -1, 0);
const _upDir = new THREE.Vector3(0, 1, 0);
const _prevUpdatePos = new THREE.Vector3();
const _losDir = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _moveDir = new THREE.Vector3();
const _birdFwd = new THREE.Vector3();
const _actionFwd = new THREE.Vector3();
const _beakReachPos = new THREE.Vector3();
const _dropPos = new THREE.Vector3();
const _dropDir = new THREE.Vector3();
const _dropHitPt = new THREE.Vector3();
const _dropDelta = new THREE.Vector3();
const _propDelta = new THREE.Vector3();
const _perchA = new THREE.Vector3();
const _perchB = new THREE.Vector3();
const _perchAB = new THREE.Vector3();
const _perchTan = new THREE.Vector3();
const _perpVec = new THREE.Vector3();
const _perchTargetScratch = new THREE.Vector3();
const _projScreen = new THREE.Vector3();
const _tapWorldDir = new THREE.Vector3();
const _beakWorld = new THREE.Vector3();
const _carryOffsetWorld = new THREE.Vector3();
const _holdQuat = new THREE.Quaternion();
const _localTarget = new THREE.Vector3();
const _rayOrigin = new THREE.Vector3();
const _probeOrigin = new THREE.Vector3();
const _hDir = new THREE.Vector3();
const _rayDir = new THREE.Vector3();
const _moveStep = new THREE.Vector3();
const _sweepDir = new THREE.Vector3();
const _ripplePos = new THREE.Vector3();
const _fxPos = new THREE.Vector3();
const _tipL = new THREE.Vector3();
const _tipR = new THREE.Vector3();
const _chirpEmitPos = new THREE.Vector3();
const _seedConsumePos = new THREE.Vector3();
const _nextDir2 = new THREE.Vector2();
const _targetHeadLook = new THREE.Vector2();
const _droppingSpawnPos = new THREE.Vector3();
const _droppingNextPos = new THREE.Vector3();
const _droppingClampedEnd = new THREE.Vector3();
const _droppingSegDir = new THREE.Vector3();
const _droppingTrailDir = new THREE.Vector3();
const _droppingUpAxis = new THREE.Vector3(0, 1, 0);
const _droppingWaterPt = new THREE.Vector3();

const _ledgeTopHit = createGroundRaycastHit();
const _ledgeProbeHit = createGroundRaycastHit();
const _dizzySideHit = createGroundRaycastHit();
const _dizzyHit = createGroundRaycastHit();
const _wallHit = createGroundRaycastHit();
const _groundHit = createGroundRaycastHit();
const _flySweepHit = createGroundRaycastHit();
const _droppingRayHit = createGroundRaycastHit();

interface ActiveDropping {
  mesh: THREE.Mesh;
  pos: THREE.Vector3;
  prevPos: THREE.Vector3;
  vel: THREE.Vector3;
  age: number;
  active: boolean;
}

/**
 * Physics and flight controller for the player sparrow (Passer domesticus).
 * - Two-footed directional passerine hopping toward screen tap coordinates with surface-specific foley & coyote time
 * - PerchSystem snapping + procedural raycast ledge auto-perching
 * - Authentic house sparrow flight: 30°–60° climb, -10° paraglider glide, screen-left/right steering & helicopter flutter-down
 * - Wall-aware prop drop & same-frame beak carry transform sync
 * - 3-stage Outdoor Dizzy Wall-Hit sequence including sunken canal water buoyancy
 */
export class SparrowController {
  public readonly sparrowMesh: SparrowMesh;
  public readonly position = new THREE.Vector3();
  public readonly velocity = new THREE.Vector3();
  public yaw = 0;
  public camera: THREE.PerspectiveCamera | null = null;

  public mode: BirdMode = 'perched';
  public dizzyState: DizzyPoseState = 'none';
  private dizzyTimer = 0;

  public carriedProp: InteractiveProp | null = null;
  public seedsGathered = 0;
  public chirpCount = 0;
  public activePerchId = '';
  public cageBounds: CageKeepInBounds | null = null;
  public windowBarrier: ClosedWindowBarrier | null = null;
  /** While true (inside indoor tutorial cage & atelier), wall/cage contacts slide without bonk/dizzy */
  public tutorialSafeZone = true;
  /** Gated false on the loading / Press Start screen so keystrokes do not move the bird prematurely */
  public inputEnabled = true;

  public flightStamina = 1.0;

  // Perch & Hermite approach transition state
  private activePerch: PerchSegment | null = null;
  private perchT = 0.5;
  private lastPerchId = '';
  private perchIgnoreTimer = 0;
  private perchTransitionActive = false;
  private perchTransElapsed = 0;
  private perchTransDuration = 0.22;
  private readonly perchTransStartOffset = new THREE.Vector3();
  private readonly perchTransStartTangent = new THREE.Vector3();
  private perchTransDist = 0;
  private perchTransStartYaw = 0;
  private perchTransYawDelta = 0;
  private perchTransStartBank = 0;
  private perchTransStartPitch = 0;
  private perchTransImpactSpeed = 0;
  private perchTransContactPlayed = false;
  private perchSettleTimer = 0;
  private perchSettleDuration = 0.22;
  private perchSettleAmp = 0;

  // Hop, Multi-Space Flight, Input Buffer, Coyote Time, Peck Lunge & Cute Bump state
  public isHopAirborne = false;
  private multiSpaceWindow = 0;
  private jumpBufferTimer = 0;
  private coyoteTimer = 0;
  private peckTimer = 0;
  private eatTimer = 0;
  private introChirpTimer = 0;
  private introLookTarget: THREE.Vector3 | null = null;
  private holdingMouseLeft = false;
  private hopPhase = 0;
  private perchHopFloatTimer = 0;
  private flapAnimTimer = 0;
  private flapCooldown = 0;
  private boundCycleTimer = 0;
  private wingTuckAmount = 0;
  private flightWingbeatTimer = 0;
  private wingFlutterFxTimer = 0;
  private helicopterDownwashTimer = 0;
  private bonkTimer = 0;
  private bonkCooldown = 0;
  private bankAngle = 0;
  private pitchAngle = 0;
  private chirpOpen = 0;
  private headLookLocal = new THREE.Vector2(0, 0);
  private idleChirpTimer = 7.5;
  private flightChirpTimer = 6.2;
  private pushChirpCooldown = 0;
  private readonly activeDroppings: ActiveDropping[] = [];
  private droppingCursor = 0;
  private droppingCooldown = 0;
  private droppingTailFlick = 0;
  private cutsceneFrozen = false;

  // Input state
  private readonly domElement: HTMLCanvasElement;
  private readonly wakeAudio: () => void;
  private keys = new Set<string>();
  private jumpQueued = false;
  private readonly queuedTapHopDir = new THREE.Vector3();
  private hasQueuedTapHopDir = false;
  private interactQueued = false;
  private chirpQueued = false;
  private mobileControls: MobileTouchControls | null = null;
  public isLongPressFlying = false;
  private flyHoldPrimed = false;
  private hopToFlyWindow = 0;
  private mouseLongPressTimer: number | null = null;
  private spaceLongPressTimer: number | null = null;
  private lastCameraYaw = 0;
  private mobileDropIgnoreGrabTimer = 0;
  private mobileWindowPushCooldown = 0;
  private readonly mobileSwitchCooldowns = new Map<string, number>();

  public npcs?: {
    handlePlayerPeck: (
      beakPos: THREE.Vector3,
      birdPos: THREE.Vector3
    ) => { hit: boolean; shooImpulse?: THREE.Vector3 };
    sampleShooGust: (playerPos: THREE.Vector3) => THREE.Vector3 | null;
    notifyTablePeck?: (beakPos: THREE.Vector3) => void;
    isHumanCollider?: (body: CANNON.Body | null | undefined) => boolean;
    checkFallingDroppingHit?: (
      prevPos: THREE.Vector3,
      nextPos: THREE.Vector3,
      birdPos: THREE.Vector3
    ) => {
      hit: boolean;
      distance: number;
      point: THREE.Vector3;
      normal: THREE.Vector3;
      attachParent?: THREE.Object3D;
    } | null;
  };

  public worldHandles?: {
    windowPullRingPos: THREE.Vector3;
    sleepyMugPos?: THREE.Vector3;
    guitarCasePos?: THREE.Vector3;
    painterEaselPos?: THREE.Vector3;
    musicBoxPos?: THREE.Vector3;
    skySanctuaryNestPos?: THREE.Vector3;
    isCageUnlatched: () => boolean;
    isWindowBarUnlatched: () => boolean;
    isWindowUnlatched: () => boolean;
    tugWindowPullCord: (pullImpulse: number) => { barUnlatched: boolean; progress: number };
    pushWindowSashes: (birdPos: THREE.Vector3, birdVel: THREE.Vector3, isPeck?: boolean) => boolean;
    applyAerodynamicWash?: (
      birdPos: THREE.Vector3,
      birdVel: THREE.Vector3,
      strength: number
    ) => void;
  };

  public onChirpCallback?: (worldPos: THREE.Vector3) => void;
  public onSeedEatenCallback?: (worldPos: THREE.Vector3, total: number) => void;
  public onPropPickedUpCallback?: (prop: InteractiveProp) => void;
  public onPropDroppedCallback?: (prop: InteractiveProp) => void;
  public getExtraAttentionTargets?: () => AttentionTarget[];

  constructor(
    scene: THREE.Scene,
    canvas: HTMLCanvasElement,
    startPos: THREE.Vector3,
    startYaw: number,
    private perches: PerchSystem,
    private physics: PhysicsWorld,
    private audio: AcousticSoundtrack,
    private fx: WorldFX,
    onFirstGesture: () => void
  ) {
    this.domElement = canvas;
    this.wakeAudio = onFirstGesture;
    this.sparrowMesh = new SparrowMesh('sparrow');
    this.position.copy(startPos);
    this.yaw = startYaw;
    this.sparrowMesh.root.position.copy(startPos);
    this.sparrowMesh.root.rotation.y = startYaw;
    scene.add(this.sparrowMesh.root);

    for (let i = 0; i < 8; i++) {
      this.activeDroppings.push({
        mesh: this.fx.createDroppingProjectileMesh(),
        pos: new THREE.Vector3(),
        prevPos: new THREE.Vector3(),
        vel: new THREE.Vector3(),
        age: 0,
        active: false,
      });
    }

    this.bindInputs();

    const initialPerch = this.perches.findNearestPerch(
      this.position,
      2.0,
      undefined,
      false,
      _perchQueryScratch
    );
    if (initialPerch) {
      this.enterPerch(initialPerch.segment, initialPerch.t, 0, true);
    } else {
      this.mode = 'grounded';
      this.coyoteTimer = 0.12;
    }
  }

  private resetHeldInputs(): void {
    this.keys.clear();
    this.holdingMouseLeft = false;
    this.isLongPressFlying = false;
    this.flyHoldPrimed = false;
    this.clearMouseLongPressTimer();
    this.clearSpaceLongPressTimer();
    this.jumpBufferTimer = 0;
  }

  private clearMouseLongPressTimer(): void {
    if (this.mouseLongPressTimer !== null) {
      window.clearTimeout(this.mouseLongPressTimer);
      this.mouseLongPressTimer = null;
    }
  }

  private clearSpaceLongPressTimer(): void {
    if (this.spaceLongPressTimer !== null) {
      window.clearTimeout(this.spaceLongPressTimer);
      this.spaceLongPressTimer = null;
    }
  }

  private isHopReadyToChainFlight(): boolean {
    return this.isHopAirborne || this.hopToFlyWindow > 0;
  }

  /**
   * Projects the sparrow's 3D world position onto screen pixels and maps the vector from the
   * sparrow to `(clientX, clientY)` onto the camera's horizontal ground plane (`_right`, `_fwd`).
   */
  private resolveScreenTapDirection(
    clientX: number,
    clientY: number,
    out: THREE.Vector3
  ): boolean {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    let birdPxX = w * 0.5;
    let birdPxY = h * 0.56;

    if (this.camera) {
      _projScreen
        .set(this.position.x, this.position.y + 0.12, this.position.z)
        .project(this.camera);
      if (_projScreen.z < 1.0) {
        birdPxX = (_projScreen.x * 0.5 + 0.5) * w;
        birdPxY = (-_projScreen.y * 0.5 + 0.5) * h;
      }
    }

    const dx = clientX - birdPxX;
    const dy = -(clientY - birdPxY);
    const dist = Math.hypot(dx, dy);
    if (dist < 10) return false;

    const nx = dx / dist;
    const ny = dy / dist;
    const sinCam = Math.sin(this.lastCameraYaw);
    const cosCam = Math.cos(this.lastCameraYaw);
    out.set(-cosCam * nx + sinCam * ny, 0, sinCam * nx + cosCam * ny);
    if (out.lengthSq() < 1e-5) return false;
    out.normalize();
    return true;
  }

  /**
   * Computes normalized horizontal turn steering (`-1` left .. `+1` right) from where the user
   * holds Long Press on screen relative to the sparrow's horizontal screen position.
   */
  private computeScreenHorizontalTurn(clientX: number): number {
    const w = Math.max(1, window.innerWidth);
    let refX = w * 0.5;
    if (this.camera) {
      _projScreen
        .set(this.position.x, this.position.y + 0.12, this.position.z)
        .project(this.camera);
      if (_projScreen.z < 1.0) {
        const birdPxX = (_projScreen.x * 0.5 + 0.5) * w;
        refX = THREE.MathUtils.clamp(birdPxX, w * 0.34, w * 0.66);
      }
    }
    const norm = THREE.MathUtils.clamp((clientX - refX) / (w * 0.35), -1, 1);
    const deadzone = 0.06;
    if (Math.abs(norm) <= deadzone) return 0;
    return Math.sign(norm) * Math.min(1, (Math.abs(norm) - deadzone) / (1 - deadzone));
  }

  private launchFromLongPressHold(): void {
    if (!this.inputEnabled || this.dizzyState !== 'none') return;
    this.flyHoldPrimed = false;
    this.isLongPressFlying = true;
    const stick = this.mobileControls?.stick;
    const screenPosTurn = stick?.active ? this.computeScreenHorizontalTurn(stick.clientX) : 0;

    if (this.mode !== 'flying') {
      const axes = this.getMoveAxes();
      _fwd.set(Math.sin(this.lastCameraYaw), 0, Math.cos(this.lastCameraYaw)).normalize();
      _right.set(-Math.cos(this.lastCameraYaw), 0, Math.sin(this.lastCameraYaw)).normalize();

      if (stick?.active) {
        const horizSteer =
          Math.abs(stick.x) > 0.12 ? stick.x : Math.abs(screenPosTurn) > 0.05 ? screenPosTurn : 0;
        if (Math.abs(horizSteer) > 0.04) {
          const launchYaw = this.lastCameraYaw - horizSteer * 0.72;
          _moveDir.set(Math.sin(launchYaw), 0, Math.cos(launchYaw)).normalize();
        } else {
          // Preserve directional tap-hop heading when holding near center without horizontal drag
          _moveDir.set(0, 0, 0);
        }
      } else {
        _moveDir.set(0, 0, 0).addScaledVector(_fwd, axes.forward).addScaledVector(_right, axes.right);
        if (_moveDir.lengthSq() > 1) _moveDir.normalize();
      }
      this.transitionHopToFlight(_moveDir);
    } else {
      // Re-engaging powered climb after gliding: immediately bank & turn toward the long-press left/right screen side
      // (unless dragging downward into straight-line helicopter flutter-down, which locks yaw without rotating)
      const isDraggingDown = Boolean(stick?.active && stick.y < -0.16);
      if (stick?.active && !isDraggingDown) {
        const horizSteer = Math.abs(stick.x) > 0.12 ? stick.x : screenPosTurn;
        if (Math.abs(horizSteer) > 0.04) {
          this.yaw -= horizSteer * 0.38;
          this.bankAngle = THREE.MathUtils.clamp(this.bankAngle + horizSteer * 0.48, -1, 1);
          const curH = Math.hypot(this.velocity.x, this.velocity.z);
          if (curH > 0.5) {
            const targetVx = Math.sin(this.yaw) * curH;
            const targetVz = Math.cos(this.yaw) * curH;
            this.velocity.x += (targetVx - this.velocity.x) * 0.55;
            this.velocity.z += (targetVz - this.velocity.z) * 0.55;
          }
        }
      }
      this.wingTuckAmount = 0;
      this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.55);
    }
  }

  public initiateGlideMechanics(): void {
    this.flyHoldPrimed = false;
    this.wingTuckAmount = 0.0;
    this.boundCycleTimer = 0.0;
    this.flapAnimTimer = 0.0;

    const fwd = this.getForwardVector(_actionFwd);
    const curHSpeed = Math.hypot(this.velocity.x, this.velocity.z);

    if (this.velocity.y > 1.2) {
      // Smoothly crest out of a 30°–60° upward climb into a buoyant -10° paraglider glide
      const bonusFwd = (this.velocity.y - 1.2) * 0.48;
      this.velocity.y = Math.min(this.velocity.y, 2.1);
      const targetH = Math.max(11.8, curHSpeed + bonusFwd);
      if (curHSpeed > 0.2) {
        const scale = targetH / curHSpeed;
        this.velocity.x *= scale;
        this.velocity.z *= scale;
      } else {
        this.velocity.x = fwd.x * targetH;
        this.velocity.z = fwd.z * targetH;
      }
    } else if (this.velocity.y < -2.4) {
      // Transition cleanly out of straight-line helicopter flutter-down into a level -10° paraglider glide
      const shedDown = Math.max(0, -this.velocity.y - 2.08);
      this.velocity.y = -2.08;
      const targetH = Math.min(15.2, Math.max(11.8, curHSpeed + shedDown * 0.16));
      if (curHSpeed > 0.2) {
        const scale = targetH / curHSpeed;
        this.velocity.x *= scale;
        this.velocity.z *= scale;
      } else {
        this.velocity.x = fwd.x * targetH;
        this.velocity.z = fwd.z * targetH;
      }
    } else if (curHSpeed < 11.6) {
      // Keep canopy pressurized at paraglider trim airspeed so releasing never stalls downward
      const targetH = 11.6;
      if (curHSpeed > 0.2) {
        const scale = targetH / curHSpeed;
        this.velocity.x *= scale;
        this.velocity.z *= scale;
      } else {
        this.velocity.x = fwd.x * targetH;
        this.velocity.z = fwd.z * targetH;
      }
    }
    this.audio.playFlightWingbeat(0.16);
  }

  public clearQueuedInputs(): void {
    this.resetHeldInputs();
    this.mobileControls?.resetAllTouches();
    this.jumpQueued = false;
    this.hasQueuedTapHopDir = false;
    this.interactQueued = false;
    this.chirpQueued = false;
  }

  public isMobileInputActive(): boolean {
    return this.mobileControls?.isMobile() ?? false;
  }

  public getMobileStick(): MobileStickState | null {
    return this.mobileControls?.stick ?? null;
  }

  public attachMobileControls(controls: MobileTouchControls): void {
    this.mobileControls = controls;
    controls.isBirdFlying = () => this.mode === 'flying';
    controls.isBirdHopReadyToFly = () => this.isHopReadyToChainFlight();
    controls.onFirstInteract = () => {
      this.wakeAudio();
    };
    controls.onFlyHoldPrime = (primed: boolean) => {
      if (!this.inputEnabled || this.dizzyState !== 'none') {
        this.flyHoldPrimed = false;
        return;
      }
      this.flyHoldPrimed = primed;
      if (primed && this.mode === 'flying') {
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.35);
      }
    };
    controls.onTapJump = (clientX: number, clientY: number) => {
      if (!this.inputEnabled) return;
      this.wakeAudio();
      this.flyHoldPrimed = false;
      if (this.dizzyState === 'grounded') {
        this.dizzyTimer = Math.max(0.08, this.dizzyTimer - 0.08);
        return;
      }
      this.hasQueuedTapHopDir = this.resolveScreenTapDirection(
        clientX,
        clientY,
        this.queuedTapHopDir
      );
      this.jumpQueued = true;
      this.jumpBufferTimer = 0.14;
    };
    controls.onLongPressStart = () => {
      if (!this.inputEnabled || this.dizzyState !== 'none') return;
      this.wakeAudio();
      this.launchFromLongPressHold();
    };
    controls.onLongPressEnd = () => {
      this.flyHoldPrimed = false;
      this.isLongPressFlying = false;
      if (this.mode === 'flying') {
        this.initiateGlideMechanics();
      }
    };
    controls.onTwoFingerTap = () => {
      if (!this.inputEnabled || this.dizzyState !== 'none' || !this.carriedProp) {
        return false;
      }
      this.mobileDropIgnoreGrabTimer = 1.35;
      this.handleBeakAction();
      return true;
    };
  }

  public consumeCarriedProp(prop: InteractiveProp): void {
    if (this.carriedProp === prop) {
      this.carriedProp = null;
      prop.carried = false;
    }
  }

  private bindInputs(): void {
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Meta' || e.metaKey) {
        this.resetHeldInputs();
        return;
      }
      if (!this.inputEnabled) return;
      this.wakeAudio();
      const code = e.code;
      if (
        code === 'Space' ||
        code === 'ArrowUp' ||
        code === 'ArrowDown' ||
        code === 'ArrowLeft' ||
        code === 'ArrowRight'
      ) {
        e.preventDefault();
      }

      if (!this.keys.has(code)) {
        if (code === 'Space') {
          if (this.mode === 'flying' || this.isHopReadyToChainFlight()) {
            // 2nd press of Tap + Long Press (or re-engaging powered flight while gliding):
            // Immediately prime wings at 0ms, and engage 30°–60° flight after 85ms hold!
            this.flyHoldPrimed = true;
            if (this.mode === 'flying') {
              this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.35);
            }
            this.clearSpaceLongPressTimer();
            this.spaceLongPressTimer = window.setTimeout(() => {
              this.spaceLongPressTimer = null;
              if (this.keys.has('Space') && this.inputEnabled && this.dizzyState === 'none') {
                this.launchFromLongPressHold();
              }
            }, 85);
          } else {
            // 1st press from rest: crisp Hop first (priming hopToFlyWindow for follow-up Long Press)
            this.jumpQueued = true;
            this.jumpBufferTimer = 0.12;
          }
        }
        if (code === 'KeyE' || code === 'KeyF') {
          this.interactQueued = true;
        }
        if (code === 'KeyQ' || code === 'KeyC' || code === 'KeyR') this.chirpQueued = true;

        if (this.dizzyState === 'grounded') {
          this.dizzyTimer = Math.max(0.08, this.dizzyTimer - 0.08);
        }
      }
      this.keys.add(code);
    });

    window.addEventListener('keyup', (e) => {
      if (e.key === 'Meta' || e.metaKey) {
        this.resetHeldInputs();
        return;
      }
      this.keys.delete(e.code);
      if (e.code === 'Space') {
        const hadPendingHoldTimer = this.spaceLongPressTimer !== null;
        this.clearSpaceLongPressTimer();
        this.flyHoldPrimed = false;
        if (this.isLongPressFlying) {
          this.isLongPressFlying = false;
          if (this.mode === 'flying') {
            this.initiateGlideMechanics();
          }
        } else if (hadPendingHoldTimer) {
          // Released before 85ms hold confirmation -> quick follow-up tap (single flap / flutter hop)
          this.jumpQueued = true;
          this.jumpBufferTimer = 0.10;
        }
      }
    });

    window.addEventListener('blur', () => {
      this.resetHeldInputs();
    });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.resetHeldInputs();
      }
    });

    this.domElement.addEventListener('mousedown', (e) => {
      if (!this.inputEnabled || this.isMobileInputActive()) return;
      this.wakeAudio();
      if (document.pointerLockElement !== this.domElement) return;
      if (e.button === 2) {
        this.chirpQueued = true;
      } else if (e.button === 0) {
        this.holdingMouseLeft = true;
        this.interactQueued = true;
        this.clearMouseLongPressTimer();
        // Pointer-locked left click only launches flight if already flying or within hop-ready window (Tap + Long Press)
        if (this.mode === 'flying' || this.isHopReadyToChainFlight()) {
          this.flyHoldPrimed = true;
          this.mouseLongPressTimer = window.setTimeout(() => {
            this.mouseLongPressTimer = null;
            if (this.holdingMouseLeft && this.inputEnabled && this.dizzyState === 'none') {
              this.launchFromLongPressHold();
            }
          }, 85);
        }
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) {
        this.holdingMouseLeft = false;
        this.flyHoldPrimed = false;
        this.clearMouseLongPressTimer();
        if (this.isLongPressFlying) {
          this.isLongPressFlying = false;
          if (this.mode === 'flying') {
            this.initiateGlideMechanics();
          }
        }
      }
    });
  }

  private getMoveAxes(): {
    forward: number;
    right: number;
    braking: boolean;
    diving: boolean;
    flutterIntensity: number;
    mobileClimb: number;
  } {
    let forward = 0;
    let right = 0;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) forward += 1;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) forward -= 1;
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) right += 1;
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) right -= 1;

    const stick = this.mobileControls?.stick;
    if (stick?.active) {
      if (this.mode === 'flying') {
        // On long press, base upward flight angle is 30° (mobileClimb = 0);
        // dragging up (stick.y > 0) smoothly increases angle up to 60° (mobileClimb = 1);
        // dragging down (stick.y < -0.16) initiates a straight-line vertical helicopter flutter-down
        // where left/right drag angle and screen X position make zero difference (sparrow never rotates).
        const isDownwardFlutter = stick.y < -0.16;
        if (isDownwardFlutter) {
          const flutterIntensity = Math.min(1, (-stick.y - 0.16) / 0.84);
          return {
            forward: 0,
            right: 0,
            braking: false,
            diving: true,
            flutterIntensity,
            mobileClimb: 0,
          };
        }

        const isUpward = stick.y > 0.01;
        const mobileClimb = isUpward ? Math.min(1, stick.y) : 0;

        // Steer toward the right/left screen position of the long press (after gliding or during climb),
        // while progressively yielding to explicit horizontal finger dragging (`|stick.x| > 0.12`).
        const screenTurn = this.computeScreenHorizontalTurn(stick.clientX) * 0.94;
        const dragAbs = Math.abs(stick.x);
        const dragBlend = dragAbs > 0.12 ? Math.min(1, (dragAbs - 0.12) / 0.32) : 0;
        const flightRight = THREE.MathUtils.clamp(
          THREE.MathUtils.lerp(screenTurn, stick.x, dragBlend),
          -1,
          1
        );

        return {
          forward: 1.0,
          right: flightRight,
          braking: false,
          diving: false,
          flutterIntensity: 0,
          mobileClimb,
        };
      }
      forward = Math.max(-1, Math.min(1, forward + stick.y));
      right = Math.max(-1, Math.min(1, right + stick.x));
    }

    const diving =
      this.keys.has('ShiftLeft') ||
      this.keys.has('ShiftRight') ||
      (this.keys.has('Space') && forward < -0.2);
    if (this.mode === 'flying' && diving) {
      return {
        forward: 0,
        right: 0,
        braking: false,
        diving: true,
        flutterIntensity: 0.92,
        mobileClimb: 0,
      };
    }

    const braking = forward < -0.2 && !this.keys.has('Space');
    const flutterIntensity = diving ? 0.92 : 0;
    const keyClimb =
      this.keys.has('Space') && (this.keys.has('ArrowUp') || this.keys.has('KeyW')) ? 1.0 : 0;
    return { forward, right, braking, diving, flutterIntensity, mobileClimb: keyClimb };
  }

  private hasBufferedJump(): boolean {
    return this.jumpQueued || this.jumpBufferTimer > 0;
  }

  private consumeBufferedJump(): void {
    this.jumpQueued = false;
    this.hasQueuedTapHopDir = false;
    this.jumpBufferTimer = 0;
  }

  public getForwardVector(out: THREE.Vector3): THREE.Vector3 {
    return out.set(Math.sin(this.yaw), 0, Math.cos(this.yaw)).normalize();
  }

  public getAirspeed(): number {
    return this.velocity.length();
  }

  public getBankAngle(): number {
    return this.bankAngle;
  }

  public getPeckPhase(): number {
    return this.peckTimer > 0 ? Math.min(1, this.peckTimer / 0.22) : 0;
  }

  public getEatPhase(): number {
    return this.eatTimer > 0 ? Math.min(1, this.eatTimer / 0.38) : 0;
  }

  public setIntroLookTarget(worldPos: THREE.Vector3 | null): void {
    if (!worldPos) {
      this.introLookTarget = null;
      return;
    }
    if (!this.introLookTarget) {
      this.introLookTarget = worldPos.clone();
    } else {
      this.introLookTarget.copy(worldPos);
    }
  }

  private computePerchAnchorPos(
    seg: PerchSegment,
    t: number,
    yaw: number,
    out: THREE.Vector3
  ): THREE.Vector3 {
    seg.getEndpoints(_perchA, _perchB);
    out.lerpVectors(_perchA, _perchB, t);
    const footFwdOffset = seg.radius >= 0.045 ? 0.105 : 0.088;
    out.x -= Math.sin(yaw) * footFwdOffset;
    out.z -= Math.cos(yaw) * footFwdOffset;
    out.y += PERCH_FOOT_CLEARANCE + (seg.getVerticalOffset ? seg.getVerticalOffset(t) : 0);
    return out;
  }

  private syncPositionToActivePerch(): void {
    if (!this.activePerch) return;
    this.computePerchAnchorPos(this.activePerch, this.perchT, this.yaw, this.position);
  }

  private enterPerch(seg: PerchSegment, t: number, impactSpeed: number, silent = false): void {
    const inVx = this.velocity.x;
    const inVy = this.velocity.y;
    const inVz = this.velocity.z;
    const prevMode = this.mode;
    const startBank = this.bankAngle;
    const startPitch = this.pitchAngle;

    this.mode = 'perched';
    this.isHopAirborne = false;
    this.coyoteTimer = 0.12;
    this.activePerch = seg;
    this.activePerchId = seg.id;
    this.perchT = t;
    this.lastPerchId = seg.id;
    this.velocity.set(0, 0, 0);
    this.hopPhase = 0;
    this.wingTuckAmount = 0;

    if (silent) {
      this.perchTransitionActive = false;
      this.perchSettleTimer = 0;
      this.bankAngle = 0;
      this.pitchAngle = 0;
      this.syncPositionToActivePerch();
      seg.onLand?.(t, impactSpeed);
      return;
    }

    const horizSpeed = Math.hypot(inVx, inVz);
    const targetYaw = horizSpeed > 0.9 ? Math.atan2(inVx, inVz) : this.yaw;
    let yawDelta = targetYaw - this.yaw;
    while (yawDelta > Math.PI) yawDelta -= Math.PI * 2;
    while (yawDelta < -Math.PI) yawDelta += Math.PI * 2;

    this.computePerchAnchorPos(seg, t, targetYaw, _perchTargetScratch);
    this.perchTransStartOffset.copy(this.position).sub(_perchTargetScratch);
    const dist = this.perchTransStartOffset.length();
    this.perchTransDist = dist;

    seg.onLand?.(t, impactSpeed);

    if (dist > 0.035 && (prevMode === 'flying' || dist > 0.12)) {
      const dur = Math.max(0.16, Math.min(0.28, 0.145 + dist * 0.068));
      const inSpeed = Math.hypot(inVx, inVy, inVz);
      if (inSpeed > 0.15) {
        const clampedS = Math.min(inSpeed, (dist / dur) * 1.45 + 1.8);
        const scale = (clampedS / inSpeed) * dur * 0.55;
        this.perchTransStartTangent.set(inVx * scale, inVy * scale, inVz * scale);
      } else {
        this.perchTransStartTangent.set(0, 0.12 * dist, 0);
      }

      this.perchTransitionActive = true;
      this.perchTransElapsed = 0;
      this.perchTransDuration = dur;
      this.perchTransStartYaw = this.yaw;
      this.perchTransYawDelta = yawDelta;
      this.perchTransStartBank = startBank;
      this.perchTransStartPitch = startPitch;
      this.perchTransImpactSpeed = impactSpeed;
      this.perchTransContactPlayed = false;
      this.perchSettleTimer = 0;
      this.flightWingbeatTimer = 0.02;
      return;
    }

    this.perchTransitionActive = false;
    this.yaw = targetYaw;
    this.bankAngle = 0;
    this.pitchAngle = 0;
    this.syncPositionToActivePerch();
    this.perchSettleDuration = 0.20;
    this.perchSettleTimer = 0.20;
    this.perchSettleAmp = Math.min(0.036, 0.015 + impactSpeed * 0.002);

    this.audio.playPerchGrip(seg.surface);
    if (Math.random() < 0.42) {
      this.chirpOpen = Math.max(this.chirpOpen, 0.65);
      this.audio.playSparrowMicroChirp('perch');
    }
    _fxPos.set(this.position.x, this.position.y - PERCH_FOOT_CLEARANCE, this.position.z);
    this.fx.spawnBurst(_fxPos, 3, false, 1.5);
  }

  private leavePerch(pushUp = true): void {
    if (this.activePerch) {
      const leavingSeg = this.activePerch;
      const leavingT = this.perchT;
      this.lastPerchId = leavingSeg.id;
      this.perchIgnoreTimer = pushUp ? 0.36 : 0.28;
      this.activePerch = null;
      this.activePerchId = '';
      leavingSeg.onLeave?.(leavingT, pushUp ? 6.5 : 3.2);
    }
    this.perchTransitionActive = false;
    this.perchSettleTimer = 0;
  }

  private detectGroundSurfaceType(pos: THREE.Vector3): SurfaceSoundType {
    if (this.checkFootSplash()) return 'water';
    if (pos.x < -37.0 && pos.y > 24.0 && pos.z > 22.0 && pos.z < 82.0) {
      return 'wood';
    }
    if (pos.x < -14.0 && pos.z < -14.0 && pos.y < 1.4) {
      return 'wood';
    }
    if (pos.x > 16.0 && pos.z > 16.0 && pos.y < 1.0) {
      return 'foliage';
    }
    return 'stone';
  }

  /**
   * Procedural dynamic ledge detector: allows auto-perching on any horizontal edge/cornice/lip
   * with an adjacent vertical drop-off, even if not explicitly registered in PerchSystem.
   */
  private tryAutoPerchOnNearbyLedge(maxSnapRadius: number): boolean {
    if (this.perchIgnoreTimer > 0 || this.dizzyState !== 'none') return false;

    const birdInAtelier = isInsideAtelier(this.position);
    const perchHit = this.perches.findNearestPerch(
      this.position,
      maxSnapRadius,
      this.lastPerchId,
      this.perchIgnoreTimer > 0,
      _perchQueryScratch
    );
    if (perchHit) {
      const perchInAtelier = isInsideAtelier(perchHit.worldPoint);
      const blockedByClosedWindow =
        Boolean(this.windowBarrier?.closed) && birdInAtelier !== perchInAtelier;
      if (!blockedByClosedWindow) {
        _rayOrigin.set(this.position.x, this.position.y + 0.16, this.position.z);
        _losDir.set(
          perchHit.worldPoint.x - _rayOrigin.x,
          perchHit.worldPoint.y + 0.35 - _rayOrigin.y,
          perchHit.worldPoint.z - _rayOrigin.z
        );
        const losLen = _losDir.length();
        let wallBlocksPerch = false;
        if (losLen > 0.25) {
          _losDir.divideScalar(losLen);
          const losHit = this.physics.raycastStaticCameraObstacle(_rayOrigin, _losDir, losLen - 0.16);
          if (losHit.hit && Math.abs(losHit.normal.y) < 0.55) {
            wallBlocksPerch = true;
          }
        }
        if (!wallBlocksPerch) {
          this.enterPerch(perchHit.segment, perchHit.t, this.velocity.length());
          return true;
        }
      }
    }

    _rayOrigin.set(this.position.x, this.position.y + 0.9, this.position.z);
    const topCheck = this.physics.raycastGround(
      _rayOrigin,
      _down,
      1.95,
      this.carriedProp?.body,
      _ledgeTopHit
    );
    if (!topCheck.hit || topCheck.normal.y < 0.78 || topCheck.point.y < 1.2) {
      return false;
    }

    let dropDirX = 0;
    let dropDirZ = 0;
    let foundDrop = false;

    for (let i = 0; i < LEDGE_PROBE_OFFSETS.length; i++) {
      const ox = LEDGE_PROBE_OFFSETS[i][0];
      const oz = LEDGE_PROBE_OFFSETS[i][1];
      const oLen = Math.hypot(ox, oz);
      _rayOrigin.set(topCheck.point.x, topCheck.point.y + 0.35, topCheck.point.z);
      _rayDir.set(ox / oLen, 0, oz / oLen);
      const wallBlock = this.physics.raycastStaticCameraObstacle(_rayOrigin, _rayDir, oLen + 0.08);
      if (wallBlock.hit && Math.abs(wallBlock.normal.y) < 0.55) {
        continue;
      }
      _probeOrigin.set(topCheck.point.x + ox, topCheck.point.y + 0.35, topCheck.point.z + oz);
      const probe = this.physics.raycastGround(
        _probeOrigin,
        _down,
        1.65,
        this.carriedProp?.body,
        _ledgeProbeHit
      );
      if (!probe.hit || probe.point.y < topCheck.point.y - 1.15) {
        dropDirX += ox;
        dropDirZ += oz;
        foundDrop = true;
      }
    }

    if (!foundDrop) return false;
    const dropLen = Math.hypot(dropDirX, dropDirZ);
    if (dropLen < 0.01) return false;
    dropDirX /= dropLen;
    dropDirZ /= dropLen;

    // Bisect along dropDir to find the exact physical lip of the horizontal surface
    let lo = 0.0;
    let hi = 0.68;
    for (let step = 0; step < 4; step++) {
      const mid = (lo + hi) * 0.5;
      _probeOrigin.set(
        topCheck.point.x + dropDirX * mid,
        topCheck.point.y + 0.35,
        topCheck.point.z + dropDirZ * mid
      );
      const probe = this.physics.raycastGround(
        _probeOrigin,
        _down,
        0.55,
        this.carriedProp?.body,
        _ledgeProbeHit
      );
      if (probe.hit && Math.abs(probe.point.y - topCheck.point.y) < 0.16) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const edgeDist = Math.max(0.0, lo - 0.035);
    const tanX = -dropDirZ;
    const tanZ = dropDirX;
    const lipX = topCheck.point.x + dropDirX * edgeDist;
    const lipZ = topCheck.point.z + dropDirZ * edgeDist;
    const lipY = topCheck.point.y;

    if (birdInAtelier) {
      const maxAutoLipX = this.windowBarrier?.closed ? ATELIER_INNER_X_MAX - 0.04 : -36.2;
      if (
        lipX < ATELIER_INNER_X_MIN + 0.1 ||
        lipX > maxAutoLipX ||
        lipZ < ATELIER_INNER_Z_MIN + 0.1 ||
        lipZ > ATELIER_INNER_Z_MAX - 0.1 ||
        lipY > ATELIER_INNER_Y_MAX - 0.45
      ) {
        return false;
      }
    }

    const a = new THREE.Vector3(lipX - tanX * 0.9, lipY, lipZ - tanZ * 0.9);
    const b = new THREE.Vector3(lipX + tanX * 0.9, lipY, lipZ + tanZ * 0.9);
    const synthId = `auto_ledge_${Math.round(lipX)}_${Math.round(lipY)}_${Math.round(lipZ)}`;
    if (synthId === this.lastPerchId && this.perchIgnoreTimer > 0) return false;

    const synthSeg: PerchSegment = {
      id: synthId,
      getEndpoints: (outA, outB) => {
        outA.copy(a);
        outB.copy(b);
      },
      radius: 0.025,
      surface: topCheck.point.x < -37.0 && topCheck.point.z > 22 ? 'wood' : 'stone',
    };

    this.enterPerch(synthSeg, 0.5, this.velocity.length());
    return true;
  }

  private triggerCuteBump(
    hitPoint: THREE.Vector3,
    hitNormal: THREE.Vector3,
    hitBody: CANNON.Body | null,
    impactSpeed: number
  ): void {
    if (this.bonkCooldown > 0 || this.tutorialSafeZone) return;
    this.bonkTimer = 0.5;
    this.bonkCooldown = 0.48;

    const meta = this.physics.getBumpMetadata(hitBody);
    const soundSurf: SurfaceSoundType =
      meta?.surface === 'stone' || meta?.surface === 'ceramic'
        ? 'wood'
        : ((meta?.surface as SurfaceSoundType | undefined) ?? 'wood');
    this.audio.playCuteBump(soundSurf);
    this.fx.spawnFeatherBurst(this.position, 4, 2.2);
    meta?.onBump?.(hitPoint, impactSpeed);

    if (this.mode === 'flying') {
      this.velocity.set(
        hitNormal.x * 5.0,
        Math.max(2.6, this.velocity.y * 0.35 + 2.8),
        hitNormal.z * 5.0
      );
    } else {
      this.velocity.x = hitNormal.x * 4.0;
      this.velocity.z = hitNormal.z * 4.0;
      this.hopPhase = 0.25;
    }
  }

  public triggerDizzyWallFall(
    hitPoint: THREE.Vector3,
    hitNormal: THREE.Vector3,
    hitBody: CANNON.Body | null,
    impactSpeed: number
  ): void {
    if (this.tutorialSafeZone || this.dizzyState !== 'none' || this.bonkCooldown > 0) return;

    this.leavePerch(false);
    this.isHopAirborne = false;
    this.coyoteTimer = 0;
    this.mode = 'flying';
    this.dizzyState = 'falling';
    this.dizzyTimer = 0;
    this.bonkCooldown = 1.6;
    this.wingTuckAmount = 0;
    this.flapAnimTimer = 0;

    if (this.carriedProp) {
      const prop = this.carriedProp;
      this.carriedProp = null;
      prop.carried = false;
      prop.body.wakeUp();
      const px = this.position.x + hitNormal.x * 0.38;
      const py = this.position.y + 0.2;
      const pz = this.position.z + hitNormal.z * 0.38;
      prop.body.position.set(px, py, pz);
      prop.body.interpolatedPosition.set(px, py, pz);
      prop.mesh.position.set(px, py, pz);
      prop.body.velocity.set(hitNormal.x * 2.8, 2.5, hitNormal.z * 2.8);
    }

    const meta = this.physics.getBumpMetadata(hitBody);
    meta?.onBump?.(hitPoint, impactSpeed);

    this.audio.playDizzyWallSmack();
    this.fx.spawnFeatherBurst(this.position, 8, 3.8);
    this.fx.attachDizzyStars(() => this.position, 6.0);

    this.position.addScaledVector(hitNormal, 0.62);
    this.velocity.set(hitNormal.x * 3.8, 1.8, hitNormal.z * 3.8);
  }

  private updateDizzySequence(dt: number): void {
    if (this.dizzyState === 'falling') {
      this.velocity.y -= 25.5 * dt;
      this.velocity.x *= Math.pow(0.45, dt);
      this.velocity.z *= Math.pow(0.45, dt);

      const hSpeed = Math.hypot(this.velocity.x, this.velocity.z);
      if (hSpeed > 0.05) {
        _hDir.set(this.velocity.x / hSpeed, 0, this.velocity.z / hSpeed);
        const sideHit = this.physics.raycastGround(
          this.position,
          _hDir,
          0.55 + hSpeed * dt,
          undefined,
          _dizzySideHit
        );
        if (sideHit.hit && sideHit.normal.y < 0.45) {
          this.velocity.x = sideHit.normal.x * 2.2;
          this.velocity.z = sideHit.normal.z * 2.2;
        }
      }

      const basin = this.physics.getBasinAtXZ(
        this.position.x,
        this.position.z,
        this.position.y
      );
      const fallDist = Math.max(0.55, -this.velocity.y * dt + 0.55);
      _rayOrigin.set(this.position.x, this.position.y + 0.25, this.position.z);
      const groundHit = this.physics.raycastGround(
        _rayOrigin,
        _down,
        fallDist,
        undefined,
        _dizzyHit
      );

      // Only land on solid ground if it is not a submerged floor beneath a water basin
      if (
        groundHit.hit &&
        groundHit.normal.y > 0.48 &&
        this.velocity.y <= 0.5 &&
        (!basin || groundHit.point.y >= basin.surfaceY - 0.05)
      ) {
        this.position.set(
          groundHit.point.x,
          groundHit.point.y + GROUND_FOOT_CLEARANCE,
          groundHit.point.z
        );
        this.velocity.set(0, 0, 0);
        this.dizzyState = 'grounded';
        this.dizzyTimer = 0.85;
        this.mode = 'grounded';
        const surf = this.detectGroundSurfaceType(this.position);
        this.audio.playLanding(9.5, surf);
        this.fx.spawnBurst(groundHit.point, 7, false, 2.8);
        return;
      }

      this.position.addScaledVector(this.velocity, dt);

      const nextBasin = this.physics.getBasinAtXZ(
        this.position.x,
        this.position.z,
        this.position.y
      );
      if (nextBasin && this.position.y <= nextBasin.surfaceY + 0.38) {
        this.position.y = nextBasin.surfaceY + 0.35;
        this.velocity.set(0, 0, 0);
        this.dizzyState = 'grounded';
        this.dizzyTimer = 0.72;
        this.mode = 'grounded';
        _ripplePos.set(this.position.x, nextBasin.surfaceY, this.position.z);
        this.fx.spawnWaterSplash(_ripplePos, 0.85);
        this.audio.playWaterPlop(this.position);
        return;
      }

      if (!nextBasin) {
        _rayOrigin.set(this.position.x, this.position.y + 1.0, this.position.z);
        const floorProbe = this.physics.raycastGround(
          _rayOrigin,
          _down,
          4.0,
          undefined,
          _dizzyHit
        );
        const minSafeY = floorProbe.hit ? floorProbe.point.y + GROUND_FOOT_CLEARANCE : GROUND_FOOT_CLEARANCE;
        if (this.position.y < minSafeY) {
          this.position.y = minSafeY;
          this.velocity.set(0, 0, 0);
          this.dizzyState = 'grounded';
          this.dizzyTimer = 0.85;
          this.mode = 'grounded';
        }
      }
      return;
    }

    if (this.dizzyState === 'grounded') {
      this.velocity.set(0, 0, 0);
      const basin = this.physics.getBasinAtXZ(this.position.x, this.position.z, this.position.y);
      if (basin && this.position.y <= basin.surfaceY + 0.45) {
        this.position.y = basin.surfaceY + 0.35;
      }
      this.dizzyTimer -= dt;
      if (this.dizzyTimer <= 0) {
        this.dizzyState = 'shaking';
        this.dizzyTimer = 0.54;
        this.fx.clearDizzyStars();
        this.audio.playFeatherRuffleShake();
        this.fx.spawnFeatherBurst(this.position, 8, 3.6);
      }
      return;
    }

    if (this.dizzyState === 'shaking') {
      this.velocity.set(0, 0, 0);
      const basin = this.physics.getBasinAtXZ(this.position.x, this.position.z, this.position.y);
      if (basin && this.position.y <= basin.surfaceY + 0.45) {
        this.position.y = basin.surfaceY + 0.35;
      }
      this.dizzyTimer -= dt;
      if (this.dizzyTimer <= 0) {
        this.dizzyState = 'none';
        this.mode = 'grounded';
        this.coyoteTimer = 0.12;
        if (basin && this.position.y <= basin.surfaceY + 0.48) {
          this.position.y = basin.surfaceY + 0.35;
        }
        this.flightStamina = 1.0;
        this.bankAngle = 0;
        this.pitchAngle = 0;
        this.bonkCooldown = 0.55;
      }
    }
  }

  /**
   * Handles the 2nd Space press of a hop: switches to full flight, or, while the cage is locked,
   * does a flapping hop that stays in 'grounded' mode.
   */
  private transitionHopToFlight(moveDir: THREE.Vector3): void {
    this.consumeBufferedJump();
    this.coyoteTimer = 0;
    const isLongPressActive =
      this.isLongPressFlying || (this.mobileControls?.isLongPressActive() ?? false);
    if (this.cageBounds?.locked && !isLongPressActive) {
      this.leavePerch(false);
      this.mode = 'grounded';
      this.isHopAirborne = true;
      this.multiSpaceWindow = 0;
      const fwd = this.getForwardVector(_birdFwd);
      const hasInput = moveDir.lengthSq() > 0.08;
      const pushDir = hasInput ? moveDir : fwd;
      const horizSpeed = hasInput ? 2.85 : 0.35;
      this.velocity.set(pushDir.x * horizSpeed, 4.35, pushDir.z * horizSpeed);
      this.hopPhase = 0.20;
      this.flapAnimTimer = 0.34;
      this.flapCooldown = 0.16;
      this.audio.playWingFlap(0.62);
      return;
    }

    this.leavePerch(true);
    this.isHopAirborne = false;
    this.hopToFlyWindow = 0;
    this.multiSpaceWindow = 0;
    this.flyHoldPrimed = false;
    this.mode = 'flying';
    const fwd = this.getForwardVector(_birdFwd);
    const hasInput = moveDir.lengthSq() > 0.08;
    const pushDir = hasInput ? moveDir : fwd;
    if (hasInput) {
      this.yaw = Math.atan2(pushDir.x, pushDir.z);
    }

    // Launch at 30° upward angle on Tap + Long Press (scaling up to 60° if already dragging upward)
    const stick = this.mobileControls?.stick;
    const keyClimb =
      this.isLongPressFlying && (this.keys.has('KeyW') || this.keys.has('ArrowUp')) ? 1.0 : 0;
    const upDrag = stick?.active ? Math.max(0, Math.min(1, stick.y)) : keyClimb;
    const climbAngleRad = THREE.MathUtils.degToRad(30 + upDrag * 30);
    const launchSpeed = this.cageBounds?.locked ? 6.8 : 11.8;
    const forwardSpeed = launchSpeed * Math.cos(climbAngleRad);
    const verticalSpeed = launchSpeed * Math.sin(climbAngleRad);

    this.velocity.set(pushDir.x * forwardSpeed, verticalSpeed, pushDir.z * forwardSpeed);
    this.pitchAngle = -climbAngleRad;
    this.flapAnimTimer = 0.85;
    this.flapCooldown = 0.14;
    this.flightWingbeatTimer = 0.10;
    this.audio.playWingFlap(0.85);
    this.worldHandles?.applyAerodynamicWash?.(this.position, this.velocity, 0.85);
    this.physics.applyWindGustAt(this.position, this.velocity, 3.6, 0.85);
    if (Math.random() < 0.58) {
      this.chirpOpen = Math.max(this.chirpOpen, 0.75);
      this.audio.playSparrowMicroChirp('takeoff');
      this.flightChirpTimer = 3.2 + Math.random() * 2.8;
    }
    this.fx.spawnBurst(this.position, 5, true, 2.1);
  }

  private applyShooImpulse(gust: THREE.Vector3): void {
    this.leavePerch(false);
    this.isHopAirborne = false;
    this.coyoteTimer = 0;
    this.mode = 'flying';
    this.bonkTimer = 0.34;
    this.flapAnimTimer = 0.65;
    this.flightWingbeatTimer = 0.135;
    this.velocity.copy(gust);
    this.audio.playWingFlap(0.95);
    this.fx.spawnFeatherBurst(this.position, 5, 2.6);
  }

  public setCutsceneFrozen(frozen: boolean): void {
    this.cutsceneFrozen = frozen;
    this.resetHeldInputs();
    this.jumpQueued = false;
    this.interactQueued = false;
    this.chirpQueued = false;

    if (frozen) {
      this.inputEnabled = false;
      this.dizzyState = 'none';
      this.dizzyTimer = 0;
      this.bonkTimer = 0;
      this.velocity.set(0, 0, 0);
      this.bankAngle = 0;
      this.pitchAngle = 0;
      this.wingTuckAmount = 0;
      this.perchTransitionActive = false;

      if (this.mode === 'flying' || this.isHopAirborne) {
        this.isHopAirborne = false;
        const nearPerch = this.perches.findNearestPerch(
          this.position,
          4.2,
          undefined,
          false,
          _perchQueryScratch
        );
        if (nearPerch) {
          this.enterPerch(nearPerch.segment, nearPerch.t, 0, true);
        } else {
          const probe = new THREE.Vector3(this.position.x, this.position.y + 0.6, this.position.z);
          const gHit = this.physics.raycastGround(probe, _down, 4.2, this.carriedProp?.body);
          const basin = this.physics.getBasinAtXZ(
            this.position.x,
            this.position.z,
            this.position.y
          );
          if (
            gHit.hit &&
            gHit.normal.y > 0.55 &&
            (!basin || gHit.point.y >= basin.surfaceY - 0.05)
          ) {
            this.position.y = gHit.point.y + GROUND_FOOT_CLEARANCE;
            this.mode = 'grounded';
            this.hopPhase = 0;
          } else {
            this.mode = 'flying';
            this.flapAnimTimer = 0.75;
          }
        }
      } else {
        this.hopPhase = 0;
      }
      this.sparrowMesh.root.position.copy(this.position);
      this.sparrowMesh.root.rotation.y = this.yaw;
      this.sparrowMesh.root.updateMatrixWorld(true);
    } else {
      this.inputEnabled = true;
      this.flightStamina = 1.0;
      if (this.mode === 'flying') {
        const fwd = this.getForwardVector(_actionFwd);
        this.velocity.set(fwd.x * 6.8, 0.8, fwd.z * 6.8);
        this.flapAnimTimer = 0.55;
      }
    }
  }

  public update(dt: number, elapsed: number, cameraYaw: number): void {
    const clampedDt = Math.min(0.05, dt);
    this.lastCameraYaw = cameraYaw;

    if (this.perchIgnoreTimer > 0) this.perchIgnoreTimer = Math.max(0, this.perchIgnoreTimer - clampedDt);
    if (this.perchSettleTimer > 0) this.perchSettleTimer = Math.max(0, this.perchSettleTimer - clampedDt);
    if (this.perchHopFloatTimer > 0) this.perchHopFloatTimer = Math.max(0, this.perchHopFloatTimer - clampedDt);
    if (this.flapCooldown > 0) this.flapCooldown = Math.max(0, this.flapCooldown - clampedDt);
    if (this.flapAnimTimer > 0) this.flapAnimTimer = Math.max(0, this.flapAnimTimer - clampedDt * 3.8);
    if (this.chirpOpen > 0) this.chirpOpen = Math.max(0, this.chirpOpen - clampedDt * 3.6);
    if (this.bonkTimer > 0) this.bonkTimer = Math.max(0, this.bonkTimer - clampedDt);
    if (this.bonkCooldown > 0) this.bonkCooldown = Math.max(0, this.bonkCooldown - clampedDt);
    if (this.multiSpaceWindow > 0) this.multiSpaceWindow = Math.max(0, this.multiSpaceWindow - clampedDt);
    if (this.hopToFlyWindow > 0) this.hopToFlyWindow = Math.max(0, this.hopToFlyWindow - clampedDt);
    if (this.jumpBufferTimer > 0) this.jumpBufferTimer = Math.max(0, this.jumpBufferTimer - clampedDt);
    if (this.peckTimer > 0) this.peckTimer = Math.max(0, this.peckTimer - clampedDt);
    if (this.eatTimer > 0) this.eatTimer = Math.max(0, this.eatTimer - clampedDt);
    if (this.introChirpTimer > 0) this.introChirpTimer = Math.max(0, this.introChirpTimer - clampedDt);
    if (this.pushChirpCooldown > 0) this.pushChirpCooldown = Math.max(0, this.pushChirpCooldown - clampedDt);
    if (this.droppingCooldown > 0) this.droppingCooldown = Math.max(0, this.droppingCooldown - clampedDt);
    if (this.droppingTailFlick > 0) this.droppingTailFlick = Math.max(0, this.droppingTailFlick - clampedDt);

    if (this.mode === 'perched' || (this.mode === 'grounded' && !this.isHopAirborne)) {
      this.coyoteTimer = 0.12;
    } else if (this.coyoteTimer > 0) {
      this.coyoteTimer = Math.max(0, this.coyoteTimer - clampedDt);
    }

    if (this.mode === 'flying') {
      this.hopToFlyWindow = 0;
    } else {
      this.flightStamina = Math.min(1.0, this.flightStamina + clampedDt * 1.45);
      this.wingTuckAmount = 0;
    }

    _prevUpdatePos.copy(this.position);

    if (this.cutsceneFrozen) {
      this.consumeBufferedJump();
      this.interactQueued = false;
      this.chirpQueued = false;
      this.velocity.set(0, 0, 0);
      this.bankAngle = 0;
      this.pitchAngle = 0;
      if (this.mode === 'flying' && this.flapAnimTimer < 0.25) {
        this.flapAnimTimer = 0.72;
      }
    } else if (this.dizzyState !== 'none') {
      this.consumeBufferedJump();
      this.interactQueued = false;
      this.chirpQueued = false;
      this.updateDizzySequence(clampedDt);
      this.enforceLockedCageBounds();
      this.enforceAtelierRoomBoundaries(_prevUpdatePos);
    } else {
      if (this.chirpQueued) {
        this.chirpQueued = false;
        this.triggerChirp();
      }

      if (this.interactQueued) {
        this.interactQueued = false;
        this.handleBeakAction();
      }

      // Autonomous idle/flight vocalizations (do not increment chirpCount, so the tutorial Q count stays exact)
      if (this.inputEnabled && !this.introLookTarget) {
        if (this.mode === 'flying') {
          this.flightChirpTimer -= clampedDt;
          if (this.flightChirpTimer <= 0) {
            this.flightChirpTimer = 6.5 + Math.random() * 6.5;
            this.chirpOpen = Math.max(this.chirpOpen, 0.82);
            if (Math.random() < 0.48) {
              this.audio.playSparrowChirp(Math.random() < 0.5);
            } else {
              this.audio.playSparrowMicroChirp('takeoff');
            }
          }
        } else {
          this.idleChirpTimer -= clampedDt;
          if (this.idleChirpTimer <= 0) {
            this.idleChirpTimer = 8.0 + Math.random() * 7.5;
            this.chirpOpen = Math.max(this.chirpOpen, 0.65);
            this.audio.playSparrowMicroChirp(this.isHopAirborne ? 'hop' : 'idle');
          }
        }
      }

      // Continuous downward pull on the Atelier window pull-cord while holding Click / E + S / Down / Shift
      if (
        this.worldHandles &&
        !this.worldHandles.isWindowBarUnlatched() &&
        (this.holdingMouseLeft || this.keys.has('KeyE') || this.keys.has('KeyF')) &&
        this.position.distanceTo(this.worldHandles.windowPullRingPos) <= 2.25
      ) {
        const pullingDown =
          this.keys.has('KeyS') ||
          this.keys.has('ArrowDown') ||
          this.keys.has('ShiftLeft') ||
          this.keys.has('ShiftRight') ||
          this.holdingMouseLeft;
        if (pullingDown) {
          this.peckTimer = Math.max(this.peckTimer, 0.12);
          const pullRate =
            this.keys.has('KeyS') || this.keys.has('ArrowDown') || this.keys.has('ShiftLeft')
              ? 9.2
              : 5.4;
          this.worldHandles.tugWindowPullCord(clampedDt * pullRate);
        }
      }

      this.checkAutoSeedNibbles();
      this.updateMobileAutoInteract(clampedDt);

      const shooGust = this.npcs?.sampleShooGust(this.position) ?? null;
      if (shooGust) {
        this.applyShooImpulse(shooGust);
      }

      const axes = this.getMoveAxes();
      _fwd.set(Math.sin(cameraYaw), 0, Math.cos(cameraYaw)).normalize();
      _right.set(-Math.cos(cameraYaw), 0, Math.sin(cameraYaw)).normalize();
      _moveDir.set(0, 0, 0).addScaledVector(_fwd, axes.forward).addScaledVector(_right, axes.right);
      if (_moveDir.lengthSq() > 1) _moveDir.normalize();

      if (this.mode === 'perched' && this.activePerch) {
        this.updatePerched(clampedDt, _moveDir);
      } else if (this.mode === 'grounded') {
        this.updateGrounded(clampedDt, _moveDir);
      } else {
        this.updateFlying(clampedDt, cameraYaw, axes);
      }

      // Continuous Sparrow body-pushing against mugs, teacups, plates, jars, pots, books & window sashes
      const birdFwd = this.getForwardVector(_birdFwd);
      const pushedProp = this.physics.pushPropsWithBirdBody(
        this.position,
        this.velocity,
        0.48,
        birdFwd
      );
      if (pushedProp) {
        const isCer =
          pushedProp.kind === 'ceramic_mug' ||
          pushedProp.kind === 'teacup' ||
          pushedProp.kind === 'porcelain_plate' ||
          pushedProp.kind === 'pigment_jar' ||
          pushedProp.kind === 'terracotta_pot';
        this.audio.playPeckTap(isCer ? 'ceramic' : 'wood', pushedProp.mesh.position);
        this.audio.addChaosImpulse(0.28);
        if (this.pushChirpCooldown <= 0 && Math.random() < 0.45) {
          this.pushChirpCooldown = 1.4;
          this.triggerMischiefChirp();
        }
      }
      this.worldHandles?.pushWindowSashes(this.position, this.velocity, false);
      if (this.mode === 'flying') {
        const airspeedSq = this.velocity.lengthSq();
        if (airspeedSq > 105.0) {
          const washStrength = Math.min(1.4, (Math.sqrt(airspeedSq) - 10.0) * 0.14) * clampedDt * 8.0;
          this.worldHandles?.applyAerodynamicWash?.(this.position, this.velocity, washStrength);
          this.physics.applyWindGustAt(this.position, this.velocity, 3.8, washStrength);
        }
      }

      this.enforceLockedCageBounds();
      this.enforceAtelierRoomBoundaries(_prevUpdatePos);
      this.jumpQueued = false;
    }

    this.sparrowMesh.root.position.copy(this.position);
    this.sparrowMesh.root.rotation.y = this.yaw;

    this.updateHeadLookIK(clampedDt);

    const axesNow = this.getMoveAxes();
    const introRoll =
      this.introChirpTimer > 0
        ? Math.sin((this.introChirpTimer / 0.56) * Math.PI) * 0.26
        : this.introLookTarget
          ? 0.12
          : 0;
    const rawEatPhase = this.getEatPhase();
    const perchGrip = this.activePerch ? (this.activePerch.radius >= 0.045 ? 1.0 : 0.14) : 0.0;
    const perchArrival = this.perchTransitionActive
      ? Math.max(0, 1.0 - this.perchTransElapsed / Math.max(0.01, this.perchTransDuration))
      : 0;
    let perchSettleOffset = 0;
    if (this.perchSettleTimer > 0 && this.perchSettleDuration > 0.01) {
      const su = 1.0 - this.perchSettleTimer / this.perchSettleDuration;
      perchSettleOffset = -this.perchSettleAmp * Math.sin(su * Math.PI) * (1.0 - su * 0.45);
    }

    let perchSlopeX = 0;
    let perchSlopeZ = 0;
    if (this.mode === 'perched' && this.activePerch) {
      this.activePerch.getEndpoints(_perchA, _perchB);
      const dx = _perchB.x - _perchA.x;
      const dy = _perchB.y - _perchA.y;
      const dz = _perchB.z - _perchA.z;
      const hLen = Math.hypot(dx, dz);
      if (hLen > 0.05 && Math.abs(dy) > 1e-4) {
        const ux = dx / hLen;
        const uz = dz / hLen;
        const m = Math.max(-0.85, Math.min(0.85, dy / hLen));
        const sinY = Math.sin(this.yaw);
        const cosY = Math.cos(this.yaw);
        perchSlopeX = m * (ux * cosY - uz * sinY);
        perchSlopeZ = m * (ux * sinY + uz * cosY);
      }
    }
    this.sparrowMesh.updatePose({
      dt: clampedDt,
      elapsed,
      mode: this.mode,
      isBraking: this.mode === 'flying' && axesNow.braking,
      isDiving: this.mode === 'flying' && axesNow.diving,
      airspeed: this.velocity.length(),
      hopAirborne: this.isHopAirborne,
      flyHoldPrime: this.flyHoldPrimed ? 1.0 : 0.0,
      wingTuck: this.wingTuckAmount,
      bonkPhase: this.bonkTimer > 0 ? this.bonkTimer / 0.5 : 0,
      dizzyState: this.dizzyState,
      dizzyPhase: this.dizzyState === 'shaking' ? 1.0 - Math.max(0, this.dizzyTimer) / 0.54 : 0,
      perchGrip,
      perchArrival,
      perchSettleOffset,
      perchSlopeX,
      perchSlopeZ,
      hopPhase: this.hopPhase,
      flapTimer: this.flapAnimTimer,
      bankAngle: this.bankAngle,
      pitchAngle: this.pitchAngle,
      chirpOpen: this.chirpOpen,
      peckPhase: this.getPeckPhase(),
      eatPhase: this.carriedProp !== null ? rawEatPhase * 0.35 : rawEatPhase,
      introHeadRoll: introRoll,
      verticalVel: this.velocity.y,
      carryingItem: this.carriedProp !== null,
      headLookLocal: this.headLookLocal,
    });

    if (this.droppingTailFlick > 0) {
      const flick = Math.sin((this.droppingTailFlick / 0.2) * Math.PI);
      this.sparrowMesh.tailPivot.rotation.x += flick * 0.34;
    }

    // Single world-matrix update per frame after pose & head IK, so held props & wingtips track the current pose without a 1-frame lag
    this.sparrowMesh.root.updateMatrixWorld(true);

    this.updateWingFlutterFX(
      clampedDt,
      this.mode === 'flying' && axesNow.diving,
      axesNow.flutterIntensity,
      this.mode === 'flying' && axesNow.braking
    );

    this.updateActiveDroppings(clampedDt);

    if (this.carriedProp) {
      if (this.carriedProp.consumed) {
        this.carriedProp = null;
      } else {
        this.sparrowMesh.beakHoldAnchor.getWorldPosition(_beakWorld);
        this.sparrowMesh.headPivot.getWorldQuaternion(_holdQuat);

        _carryOffsetWorld.copy(this.carriedProp.carryOffset).applyQuaternion(_holdQuat);
        _beakWorld.add(_carryOffsetWorld);

        this.carriedProp.mesh.position.copy(_beakWorld);
        this.carriedProp.mesh.quaternion.copy(_holdQuat);
        this.carriedProp.body.position.set(_beakWorld.x, _beakWorld.y, _beakWorld.z);
        this.carriedProp.body.interpolatedPosition.set(_beakWorld.x, _beakWorld.y, _beakWorld.z);
        this.carriedProp.body.quaternion.set(_holdQuat.x, _holdQuat.y, _holdQuat.z, _holdQuat.w);
        this.carriedProp.body.interpolatedQuaternion.set(
          _holdQuat.x,
          _holdQuat.y,
          _holdQuat.z,
          _holdQuat.w
        );
        this.carriedProp.body.velocity.set(0, 0, 0);
        this.carriedProp.body.angularVelocity.set(0, 0, 0);
      }
    }
  }

  private enforceLockedCageBounds(): void {
    const cb = this.cageBounds;
    if (!cb || !cb.locked) return;

    const maxCeiling = Math.min(cb.ceilingY, 38.25);
    if (this.position.y > maxCeiling) {
      this.position.y = maxCeiling;
      if (this.velocity.y > 0) this.velocity.y = 0;
    }

    const domeFactor =
      this.position.y > 37.3
        ? Math.cos(Math.min(1.2, ((this.position.y - 37.3) / 2.2) * Math.PI * 0.5))
        : 1.0;
    const effectiveRadius = Math.max(0.85, cb.radius * domeFactor);

    const dx = this.position.x - cb.center.x;
    const dz = this.position.z - cb.center.z;
    const r = Math.hypot(dx, dz);
    if (r > effectiveRadius && r > 0.0001) {
      const nx = -dx / r;
      const nz = -dz / r;
      this.position.x = cb.center.x + (dx / r) * effectiveRadius;
      this.position.z = cb.center.z + (dz / r) * effectiveRadius;
      const vn = this.velocity.x * nx + this.velocity.z * nz;
      if (vn < 0) {
        this.velocity.x -= nx * vn;
        this.velocity.z -= nz * vn;
      }
    }

    if (this.position.y < cb.floorY) {
      const wasHopAirborne = this.isHopAirborne;
      this.position.y = cb.floorY;
      if (this.velocity.y < 0) this.velocity.y = 0;
      this.isHopAirborne = false;
      this.perchHopFloatTimer = 0;
      if (wasHopAirborne) {
        this.hopPhase = 0;
        this.perchSettleDuration = 0.24;
        this.perchSettleTimer = 0.24;
        this.perchSettleAmp = 0.028;
        this.audio.playHop(this.detectGroundSurfaceType(this.position));
      }
      if (this.mode === 'flying') this.mode = 'grounded';
    }
  }

  private isInsideBirdcage(pos: THREE.Vector3): boolean {
    const cb = this.cageBounds;
    if (!cb) return false;
    const dx = pos.x - cb.center.x;
    const dz = pos.z - cb.center.z;
    return (
      dx * dx + dz * dz <= (cb.radius + 0.45) * (cb.radius + 0.45) &&
      pos.y >= cb.floorY - 0.35 &&
      pos.y <= cb.ceilingY + 0.35
    );
  }

  /**
   * Watertight analytic boundary enforcement for the 2nd-story Conservatory Atelier
   * (`NW Block`, `x: [-88, -38], y: [26.2, 51.6], z: [24, 80]`) and its East hinged French
   * casement window aperture (`z: [43.95, 60.05], y: [31.38, 47.25]`), eliminating any
   * shallow-angle, corner, perch-snap, or frame-rate spike wall clipping.
   */
  private enforceAtelierRoomBoundaries(prevPos: THREE.Vector3): void {
    const windowClosed = Boolean(this.windowBarrier?.closed);
    const inOpenEastTunnel =
      !windowClosed &&
      prevPos.x > -39.3 &&
      prevPos.z >= ATELIER_APERTURE_Z_MIN - 0.15 &&
      prevPos.z <= ATELIER_APERTURE_Z_MAX + 0.15 &&
      prevPos.y >= ATELIER_APERTURE_Y_MIN - 0.15 &&
      prevPos.y <= ATELIER_APERTURE_Y_MAX + 0.15;
    const prevInRoom =
      !inOpenEastTunnel &&
      prevPos.x >= -87.8 &&
      prevPos.x <= -38.85 &&
      prevPos.y >= 25.5 &&
      prevPos.y <= 52.3 &&
      prevPos.z >= 24.3 &&
      prevPos.z <= 79.7;

    let clampedDistSq = 0;
    const beforeX = this.position.x;
    const beforeY = this.position.y;
    const beforeZ = this.position.z;

    if (prevInRoom) {
      if (this.position.x < ATELIER_INNER_X_MIN) {
        this.position.x = ATELIER_INNER_X_MIN;
        if (this.velocity.x < 0) this.velocity.x = 0;
      }
      if (this.position.z < ATELIER_INNER_Z_MIN) {
        this.position.z = ATELIER_INNER_Z_MIN;
        if (this.velocity.z < 0) this.velocity.z = 0;
      } else if (this.position.z > ATELIER_INNER_Z_MAX) {
        this.position.z = ATELIER_INNER_Z_MAX;
        if (this.velocity.z > 0) this.velocity.z = 0;
      }
      if (this.position.y < ATELIER_INNER_Y_MIN) {
        this.position.y = ATELIER_INNER_Y_MIN;
        if (this.velocity.y < 0) this.velocity.y = 0;
      } else if (this.position.y > ATELIER_INNER_Y_MAX) {
        this.position.y = ATELIER_INNER_Y_MAX;
        if (this.velocity.y > 0) this.velocity.y = 0;
      }

      // South plaster chimney breast (`x: [-67.6, -56.4], y >= 35.8, front z = 26.725`)
      // & walnut regulator clock (`x: [-63.83, -60.17], y: [37.8, 44.9], front z = 27.45`)
      if (
        this.position.y >= 35.30 &&
        this.position.x > -68.22 &&
        this.position.x < -55.78 &&
        this.position.z < 27.36
      ) {
        if (prevPos.x <= -68.02 && prevPos.z < 27.25) {
          this.position.x = -68.22;
          if (this.velocity.x > 0) this.velocity.x = 0;
        } else if (prevPos.x >= -55.98 && prevPos.z < 27.25) {
          this.position.x = -55.78;
          if (this.velocity.x < 0) this.velocity.x = 0;
        } else {
          this.position.z = 27.36;
          if (this.velocity.z < 0) this.velocity.z = 0;
        }
      }
      if (
        this.position.y >= 37.35 &&
        this.position.y <= 45.35 &&
        this.position.x > -64.38 &&
        this.position.x < -59.62 &&
        this.position.z < 28.02
      ) {
        if (prevPos.x <= -64.2 && prevPos.z < 27.9) {
          this.position.x = -64.38;
          if (this.velocity.x > 0) this.velocity.x = 0;
        } else if (prevPos.x >= -59.8 && prevPos.z < 27.9) {
          this.position.x = -59.62;
          if (this.velocity.x < 0) this.velocity.x = 0;
        } else {
          this.position.z = 28.02;
          if (this.velocity.z < 0) this.velocity.z = 0;
        }
      }

      // Solid East-South & East-North wall piers + pleated velvet drapes below the `y = 48.99` curtain rod
      if (this.position.y < 48.55) {
        const inDrapeColumn =
          (this.position.z >= 40.3 && this.position.z < ATELIER_APERTURE_Z_MIN) ||
          (this.position.z > ATELIER_APERTURE_Z_MAX && this.position.z <= 63.7);
        if (inDrapeColumn && this.position.x > -40.76) {
          this.position.x = -40.76;
          if (this.velocity.x > 0) this.velocity.x = 0;
        } else if (
          (this.position.z < ATELIER_APERTURE_Z_MIN || this.position.z > ATELIER_APERTURE_Z_MAX) &&
          this.position.x > -40.42
        ) {
          this.position.x = -40.42;
          if (this.velocity.x > 0) this.velocity.x = 0;
        }
      }

      if (this.position.x > ATELIER_INNER_X_MAX) {
        const alignedWithAperture =
          !windowClosed &&
          this.position.z >= ATELIER_APERTURE_Z_MIN &&
          this.position.z <= ATELIER_APERTURE_Z_MAX &&
          this.position.y >= ATELIER_APERTURE_Y_MIN &&
          this.position.y <= ATELIER_APERTURE_Y_MAX &&
          prevPos.z >= ATELIER_APERTURE_Z_MIN - 0.85 &&
          prevPos.z <= ATELIER_APERTURE_Z_MAX + 0.85 &&
          prevPos.y >= ATELIER_APERTURE_Y_MIN - 0.85 &&
          prevPos.y <= ATELIER_APERTURE_Y_MAX + 0.85;

        if (!alignedWithAperture) {
          this.position.x = ATELIER_INNER_X_MAX;
          if (this.velocity.x > 0) this.velocity.x = 0;
        }
      }
    }

    // East French window stone frame tunnel (`x` between `-39.95` and `-37.45`):
    // prevent clipping sideways into stone jambs, downward into sill wall, or upward into lintel
    if (
      this.position.x > ATELIER_INNER_X_MAX &&
      this.position.x < ATELIER_OUTER_X_EAST &&
      this.position.y >= 25.5 &&
      this.position.y <= 52.3 &&
      this.position.z >= 23.8 &&
      this.position.z <= 80.2
    ) {
      if (windowClosed) {
        if (prevPos.x <= -38.7) {
          this.position.x = ATELIER_INNER_X_MAX;
          if (this.velocity.x > 0) this.velocity.x = 0;
        } else {
          this.position.x = ATELIER_OUTER_X_EAST;
          if (this.velocity.x < 0) this.velocity.x = 0;
        }
      } else {
        // If approaching from outdoors against the solid East-North or East-South wall piers/lintel/sill, stop outside
        const prevOutsideEast = prevPos.x >= ATELIER_OUTER_X_EAST - 0.08;
        const outsideAimedAtWall =
          prevOutsideEast &&
          (prevPos.z < ATELIER_APERTURE_Z_MIN - 0.65 ||
            prevPos.z > ATELIER_APERTURE_Z_MAX + 0.65 ||
            prevPos.y < ATELIER_APERTURE_Y_MIN - 0.65 ||
            prevPos.y > ATELIER_APERTURE_Y_MAX + 0.65);

        if (outsideAimedAtWall) {
          this.position.x = ATELIER_OUTER_X_EAST;
          if (this.velocity.x < 0) this.velocity.x = 0;
        } else {
          if (this.position.z < ATELIER_APERTURE_Z_MIN) {
            this.position.z = ATELIER_APERTURE_Z_MIN;
            if (this.velocity.z < 0) this.velocity.z = 0;
          } else if (this.position.z > ATELIER_APERTURE_Z_MAX) {
            this.position.z = ATELIER_APERTURE_Z_MAX;
            if (this.velocity.z > 0) this.velocity.z = 0;
          }
          if (this.position.y < ATELIER_APERTURE_Y_MIN) {
            this.position.y = ATELIER_APERTURE_Y_MIN;
            if (this.velocity.y < 0) this.velocity.y = 0;
          } else if (this.position.y > ATELIER_APERTURE_Y_MAX) {
            this.position.y = ATELIER_APERTURE_Y_MAX;
            if (this.velocity.y > 0) this.velocity.y = 0;
          }
        }
      }
    }

    // Exterior solid facade guard on West, North, and South Atelier walls (`y: [25.6, 52.2]`)
    if (
      !prevInRoom &&
      this.position.y >= 25.6 &&
      this.position.y <= 52.2 &&
      this.position.x > -88.52 &&
      this.position.x < ATELIER_OUTER_X_EAST &&
      this.position.z > 23.48 &&
      this.position.z < 80.52
    ) {
      if (prevPos.x <= -88.0) {
        this.position.x = -88.52;
        if (this.velocity.x > 0) this.velocity.x = 0;
      } else if (prevPos.z <= 24.0) {
        this.position.z = 23.48;
        if (this.velocity.z > 0) this.velocity.z = 0;
      } else if (prevPos.z >= 80.0) {
        this.position.z = 80.52;
        if (this.velocity.z < 0) this.velocity.z = 0;
      }
    }

    const dx = this.position.x - beforeX;
    const dy = this.position.y - beforeY;
    const dz = this.position.z - beforeZ;
    clampedDistSq = dx * dx + dy * dy + dz * dz;
    if (clampedDistSq > 0.08 * 0.08 && this.mode === 'perched') {
      this.leavePerch(false);
      this.mode = 'grounded';
    }
  }

  private updatePerched(dt: number, moveDir: THREE.Vector3): void {
    const seg = this.activePerch!;
    seg.getEndpoints(_perchA, _perchB);
    _perchAB.subVectors(_perchB, _perchA);
    const segLen = Math.max(0.01, _perchAB.length());
    _perchTan.copy(_perchAB).divideScalar(segLen);

    if (this.hasBufferedJump()) {
      // Sparrow must always HOP first from a perch before flying (Tap = directional Hop toward tap, Tap + Long Press = Fly)
      const hadTapDir = this.hasQueuedTapHopDir;
      if (hadTapDir) _tapWorldDir.copy(this.queuedTapHopDir);
      this.consumeBufferedJump();
      this.coyoteTimer = 0;
      const inCage = this.isInsideBirdcage(this.position);
      this.leavePerch(false);
      this.perchIgnoreTimer = Math.max(this.perchIgnoreTimer, inCage ? 0.45 : 0.34);
      this.mode = 'grounded';
      this.isHopAirborne = true;
      this.hopToFlyWindow = 0.72;
      this.multiSpaceWindow = 0.72;
      this.perchHopFloatTimer = inCage ? 0.75 : isInsideAtelier(this.position) ? 0.44 : 0.30;
      const indoors = isInsideAtelier(this.position);
      const fwd = this.getForwardVector(_birdFwd);
      const hasMoveInput = moveDir.lengthSq() > 0.08;
      const dir = hadTapDir ? _tapWorldDir : hasMoveInput ? moveDir : fwd;
      if (hadTapDir || hasMoveInput) {
        this.yaw = Math.atan2(dir.x, dir.z);
      }
      const hopHoriz = hadTapDir
        ? inCage
          ? 2.25
          : indoors
            ? 3.25
            : 4.55
        : hasMoveInput
          ? inCage
            ? 2.15
            : indoors
              ? 2.85
              : 4.2
          : inCage
            ? 1.35
            : 0.95;
      this.velocity.set(
        dir.x * hopHoriz,
        inCage ? 2.35 : indoors ? 4.4 : 5.2,
        dir.z * hopHoriz
      );
      this.hopPhase = 0.20; // Skip crouch phase for instant liftoff reaction
      this.flapAnimTimer = Math.max(this.flapAnimTimer, inCage ? 0.42 : 0.24);
      this.audio.playFlightWingbeat(0.22);
      this.audio.playHop(seg.surface);
      return;
    }

    if (this.perchTransitionActive) {
      this.perchTransElapsed += dt;
      const u = Math.min(1.0, this.perchTransElapsed / Math.max(0.01, this.perchTransDuration));
      const s = u * u * (3.0 - 2.0 * u);
      const endYaw = this.perchTransStartYaw + this.perchTransYawDelta;

      this.yaw = this.perchTransStartYaw + this.perchTransYawDelta * s;
      this.bankAngle = this.perchTransStartBank * (1.0 - s);
      this.pitchAngle = this.perchTransStartPitch * (1.0 - s) - 0.36 * Math.sin(u * Math.PI);

      this.computePerchAnchorPos(seg, this.perchT, endYaw, this.position);
      const u2 = u * u;
      const u3 = u2 * u;
      const h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
      const h10 = u3 - 2.0 * u2 + u;
      const arcLift = Math.sin(u * Math.PI) * Math.min(0.20, 0.025 + this.perchTransDist * 0.12);
      this.position.x += this.perchTransStartOffset.x * h00 + this.perchTransStartTangent.x * h10;
      this.position.y +=
        this.perchTransStartOffset.y * h00 + this.perchTransStartTangent.y * h10 + arcLift;
      this.position.z += this.perchTransStartOffset.z * h00 + this.perchTransStartTangent.z * h10;

      if (u < 0.68) {
        this.flightWingbeatTimer -= dt;
        if (this.flightWingbeatTimer <= 0) {
          this.flightWingbeatTimer = 0.085;
          this.audio.playFlightWingbeat(0.36 * (1.0 - u * 0.7));
        }
      }

      if (u >= 0.84 && !this.perchTransContactPlayed) {
        this.perchTransContactPlayed = true;
        this.audio.playPerchGrip(seg.surface);
        if (Math.random() < 0.42) {
          this.chirpOpen = Math.max(this.chirpOpen, 0.65);
          this.audio.playSparrowMicroChirp('perch');
        }
        _fxPos.set(this.position.x, this.position.y - PERCH_FOOT_CLEARANCE, this.position.z);
        this.fx.spawnBurst(_fxPos, 3, false, 1.5);
      }

      if (u >= 1.0) {
        this.perchTransitionActive = false;
        this.yaw = endYaw;
        this.bankAngle = 0;
        this.pitchAngle = 0;
        this.perchSettleDuration = 0.22;
        this.perchSettleTimer = 0.22;
        this.perchSettleAmp = Math.min(0.038, 0.016 + this.perchTransImpactSpeed * 0.0022);
        this.syncPositionToActivePerch();
      }
      return;
    }

    if (moveDir.lengthSq() > 0.08) {
      const along = moveDir.dot(_perchTan);
      _perpVec.copy(moveDir).addScaledVector(_perchTan, -along);
      const perpMag = _perpVec.length();

      if (perpMag > 0.72 && Math.abs(along) < 0.55) {
        const inCage = this.isInsideBirdcage(this.position);
        this.leavePerch(false);
        this.coyoteTimer = 0.12;
        _rayOrigin.set(
          this.position.x + moveDir.x * 0.45,
          this.position.y + 0.25,
          this.position.z + moveDir.z * 0.45
        );
        const dropCheck = this.physics.raycastGround(
          _rayOrigin,
          _down,
          2.25,
          this.carriedProp?.body,
          _groundHit
        );
        const dropBasin = this.physics.getBasinAtXZ(
          _rayOrigin.x,
          _rayOrigin.z,
          this.position.y
        );
        const indoors = isInsideAtelier(this.position);
        const targetGroundY = dropCheck.hit ? dropCheck.point.y + GROUND_FOOT_CLEARANCE : -Infinity;
        const dropHeight = this.position.y - targetGroundY;
        if (
          dropCheck.hit &&
          dropCheck.normal.y > 0.52 &&
          (!dropBasin || dropCheck.point.y >= dropBasin.surfaceY - 0.05) &&
          dropHeight <= 0.24
        ) {
          this.mode = 'grounded';
          _rayOrigin.set(this.position.x, this.position.y + 0.14, this.position.z);
          _rayDir.set(moveDir.x, 0, moveDir.z);
          const stepClear = this.physics.raycastStaticCameraObstacle(_rayOrigin, _rayDir, 0.94);
          const safeStep =
            stepClear.hit && Math.abs(stepClear.normal.y) < 0.5
              ? Math.max(0, Math.min(0.38, stepClear.distance - 0.54))
              : 0.38;
          this.position.x += moveDir.x * safeStep;
          this.position.z += moveDir.z * safeStep;
          this.position.y = targetGroundY;
          this.hopPhase = 0.22;
          const stepOffSpeed = indoors ? 4.25 : 6.4;
          this.velocity.set(moveDir.x * stepOffSpeed, 0, moveDir.z * stepOffSpeed);
          this.audio.playHop(this.detectGroundSurfaceType(this.position));
        } else {
          // Graceful flutter-cushioned hop arc off elevated perches (such as the starting birdcage branch)
          this.perchIgnoreTimer = Math.max(this.perchIgnoreTimer, inCage ? 0.45 : 0.34);
          this.mode = 'grounded';
          this.isHopAirborne = true;
          this.multiSpaceWindow = 0.42;
          this.perchHopFloatTimer = inCage ? 0.75 : indoors ? 0.48 : 0.34;
          const hopFwdSpeed = inCage ? 2.15 : indoors ? 2.95 : 4.2;
          const hopUpSpeed = inCage ? 1.85 : indoors ? 2.25 : 2.55;
          this.velocity.set(moveDir.x * hopFwdSpeed, hopUpSpeed, moveDir.z * hopFwdSpeed);
          this.hopPhase = 0.14;
          this.flapAnimTimer = Math.max(this.flapAnimTimer, inCage ? 0.42 : 0.32);
          this.audio.playFlightWingbeat(0.24);
        }
        return;
      }

      const indoorsPerch = isInsideAtelier(this.position);
      const perchSlideSpeed = indoorsPerch ? 3.0 : 4.4;
      const perchHopCadence = indoorsPerch ? 4.1 : 5.6;
      this.perchT = Math.max(
        0.05,
        Math.min(0.95, this.perchT + (along * perchSlideSpeed * dt) / segLen)
      );
      this.hopPhase = (this.hopPhase + dt * perchHopCadence) % 1.0;
      if (this.hopPhase < dt * perchHopCadence) {
        this.audio.playHop(seg.surface);
      }

      const desiredYaw = Math.atan2(moveDir.x, moveDir.z);
      let dYaw = desiredYaw - this.yaw;
      while (dYaw > Math.PI) dYaw -= Math.PI * 2;
      while (dYaw < -Math.PI) dYaw += Math.PI * 2;
      this.yaw += dYaw * Math.min(1, dt * 8.5);
    } else {
      this.hopPhase = 0;
    }

    this.syncPositionToActivePerch();
  }

  private updateGrounded(dt: number, moveDir: THREE.Vector3): void {
    const indoors = isInsideAtelier(this.position);
    if (this.hasBufferedJump()) {
      const hadTapDir = this.hasQueuedTapHopDir;
      if (hadTapDir) _tapWorldDir.copy(this.queuedTapHopDir);
      this.consumeBufferedJump();
      this.coyoteTimer = 0;
      const inCage = this.isInsideBirdcage(this.position);
      const wasAlreadyHopReady = this.isHopAirborne || this.hopToFlyWindow > 0;
      this.isHopAirborne = true;
      this.hopToFlyWindow = 0.72;
      this.multiSpaceWindow = 0.72;
      const fwd = this.getForwardVector(_birdFwd);
      const hasMoveInput = moveDir.lengthSq() > 0.05;
      const dir = hadTapDir ? _tapWorldDir : hasMoveInput ? moveDir : fwd;
      if (hadTapDir || hasMoveInput) {
        this.yaw = Math.atan2(dir.x, dir.z);
      }
      const hopHoriz = hadTapDir
        ? inCage
          ? 2.35
          : indoors
            ? 3.55
            : 4.95
        : hasMoveInput
          ? indoors
            ? 3.1
            : 4.6
          : wasAlreadyHopReady
            ? 1.6
            : 0.65;
      this.velocity.set(
        dir.x * hopHoriz,
        wasAlreadyHopReady ? (indoors ? 4.5 : 5.2) : indoors ? 4.9 : 5.7,
        dir.z * hopHoriz
      );
      this.hopPhase = 0.20; // Instant liftoff frame ready to chain into flight with 0ms delay
      this.flapAnimTimer = Math.max(this.flapAnimTimer, wasAlreadyHopReady ? 0.30 : 0.22);
      if (wasAlreadyHopReady) {
        this.audio.playWingFlap(0.55);
      } else {
        this.audio.playHop(this.detectGroundSurfaceType(this.position));
      }
      _fxPos.set(this.position.x, this.position.y - 0.34, this.position.z);
      this.fx.spawnBurst(_fxPos, 3, false, 1.5);
    }

    if (!this.isHopAirborne) {
      const moveLen = Math.min(1, moveDir.length());
      const baseHopSpeed = indoors
        ? this.carriedProp
          ? 4.35 - this.carriedProp.beakWeight * 0.9
          : 4.85
        : this.carriedProp
          ? 6.8 - this.carriedProp.beakWeight * 1.4
          : 7.8;
      const hopSpeed = baseHopSpeed * (0.48 + 0.52 * moveLen);
      const hopCadence = (indoors ? 3.9 : 5.4) * (0.74 + 0.26 * moveLen);
      if (moveLen > 0.16 && this.bonkTimer <= 0.25) {
        const invLen = 1.0 / moveLen;
        const dirX = moveDir.x * invLen;
        const dirZ = moveDir.z * invLen;
        const prevPhase = this.hopPhase;
        this.hopPhase = (this.hopPhase + dt * hopCadence) % 1.0;
        if (this.hopPhase < prevPhase) {
          this.audio.playHop(this.detectGroundSurfaceType(this.position));
        }

        const desiredYaw = Math.atan2(dirX, dirZ);
        let diff = desiredYaw - this.yaw;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        this.yaw += diff * Math.min(1, dt * 12.0);

        const surge = 0.65 + Math.sin(this.hopPhase * Math.PI) * 0.65;
        this.velocity.x = dirX * hopSpeed * surge;
        this.velocity.z = dirZ * hopSpeed * surge;
      } else if (this.bonkTimer <= 0.25) {
        this.hopPhase = Math.max(0, this.hopPhase - dt * 6.0);
        this.velocity.x *= Math.pow(0.001, dt);
        this.velocity.z *= Math.pow(0.001, dt);
      }
    } else {
      const inCageAir = this.isInsideBirdcage(this.position);
      const floatBlend = Math.min(1, this.perchHopFloatTimer * 2.5);
      this.hopPhase = Math.min(0.95, this.hopPhase + dt * (floatBlend > 0.05 ? 1.55 : 2.15));
      if (moveDir.lengthSq() > 0.04) {
        const desiredYaw = Math.atan2(moveDir.x, moveDir.z);
        let diff = desiredYaw - this.yaw;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        this.yaw += diff * Math.min(1, dt * 9.0);
        const airHopSteer = inCageAir ? 2.25 : indoors ? 3.4 : 5.2;
        const steerRate = floatBlend > 0.05 ? 3.8 : 5.5;
        this.velocity.x += (moveDir.x * airHopSteer - this.velocity.x) * Math.min(1, dt * steerRate);
        this.velocity.z += (moveDir.z * airHopSteer - this.velocity.z) * Math.min(1, dt * steerRate);
      }
      if (floatBlend > 0.05) {
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.22 * floatBlend);
      }
      if (this.flyHoldPrimed) {
        // Zero-delay hover cushion during the 85ms follow-up press confirmation so the bird never drops before flying
        this.velocity.y = Math.max(this.velocity.y, 2.2);
      } else {
        const effectiveGrav = 24.0 - floatBlend * 12.6;
        this.velocity.y -= effectiveGrav * dt;
        if (floatBlend > 0.05) {
          const terminalFall = inCageAir ? -4.15 : -5.6;
          if (this.velocity.y < terminalFall) {
            this.velocity.y += (terminalFall - this.velocity.y) * Math.min(1, dt * 11.5 * floatBlend);
          }
        }
      }
    }

    // Multi-ray horizontal obstacle sweep + 2nd-pass corner clearance while grounded
    const hSpeed = Math.hypot(this.velocity.x, this.velocity.z);
    if (hSpeed > 0.05) {
      _hDir.set(this.velocity.x / hSpeed, 0, this.velocity.z / hSpeed);
      for (let i = 0; i < GROUND_WALL_ANGLES.length; i++) {
        const ang = GROUND_WALL_ANGLES[i];
        const cosA = Math.cos(ang);
        const sinA = Math.sin(ang);
        _rayDir.set(_hDir.x * cosA - _hDir.z * sinA, 0, _hDir.x * sinA + _hDir.z * cosA);
        _rayOrigin.set(this.position.x, this.position.y + 0.12, this.position.z);
        const reach = 0.58 + hSpeed * dt;
        const wallHit = this.physics.raycastGround(
          _rayOrigin,
          _rayDir,
          reach,
          this.carriedProp?.body,
          _wallHit
        );
        if (wallHit.hit && wallHit.normal.y < 0.45) {
          if (wallHit.distance < 0.54) {
            this.position.addScaledVector(wallHit.normal, 0.54 - wallHit.distance);
          }
          const vn = this.velocity.x * wallHit.normal.x + this.velocity.z * wallHit.normal.z;
          if (vn < 0) {
            const meta = this.physics.getBumpMetadata(wallHit.body);
            if (!this.tutorialSafeZone && ang === 0 && -vn > 5.4 && meta) {
              this.triggerCuteBump(wallHit.point, wallHit.normal, wallHit.body, -vn);
              break;
            }
            this.velocity.x -= wallHit.normal.x * vn;
            this.velocity.z -= wallHit.normal.z * vn;
          }
        }
        // Static-only ray ensures dynamic clutter near walls never masks the structural wall behind it
        const staticHit = this.physics.raycastStaticCameraObstacle(_rayOrigin, _rayDir, reach);
        if (staticHit.hit && Math.abs(staticHit.normal.y) < 0.45) {
          if (staticHit.distance < 0.54) {
            this.position.x += staticHit.normal.x * (0.54 - staticHit.distance);
            this.position.z += staticHit.normal.z * (0.54 - staticHit.distance);
          }
          const svn = this.velocity.x * staticHit.normal.x + this.velocity.z * staticHit.normal.z;
          if (svn < 0) {
            this.velocity.x -= staticHit.normal.x * svn;
            this.velocity.z -= staticHit.normal.z * svn;
          }
        }
      }

      // 2nd-pass post-deflection corner check so sliding along Wall 1 never steps through Wall 2
      const postHSpeed = Math.hypot(this.velocity.x, this.velocity.z);
      if (postHSpeed > 0.02) {
        _rayDir.set(this.velocity.x / postHSpeed, 0, this.velocity.z / postHSpeed);
        _rayOrigin.set(this.position.x, this.position.y + 0.12, this.position.z);
        const cornerHit = this.physics.raycastStaticCameraObstacle(
          _rayOrigin,
          _rayDir,
          0.56 + postHSpeed * dt
        );
        if (cornerHit.hit && Math.abs(cornerHit.normal.y) < 0.45) {
          if (cornerHit.distance < 0.54) {
            this.position.x += cornerHit.normal.x * (0.54 - cornerHit.distance);
            this.position.z += cornerHit.normal.z * (0.54 - cornerHit.distance);
          }
          const cvn = this.velocity.x * cornerHit.normal.x + this.velocity.z * cornerHit.normal.z;
          if (cvn < 0) {
            this.velocity.x -= cornerHit.normal.x * cvn;
            this.velocity.z -= cornerHit.normal.z * cvn;
          }
        }
      }
    }

    this.position.x += this.velocity.x * dt;
    this.position.z += this.velocity.z * dt;

    const waterBasinAtXZ = this.physics.getBasinAtXZ(
      this.position.x,
      this.position.z,
      this.position.y
    );

    if (this.isHopAirborne) {
      if (this.velocity.y > 0) {
        _rayOrigin.set(this.position.x, this.position.y, this.position.z);
        const ceilHit = this.physics.raycastStaticCameraObstacle(
          _rayOrigin,
          _upDir,
          0.54 + this.velocity.y * dt
        );
        if (ceilHit.hit && ceilHit.normal.y < -0.45) {
          if (ceilHit.distance < 0.52) {
            this.position.y -= 0.52 - ceilHit.distance;
          }
          this.velocity.y = 0;
        }
      }
      this.position.y += this.velocity.y * dt;
      _rayOrigin.set(this.position.x, this.position.y + 0.55, this.position.z);
      const groundHit = this.physics.raycastGround(
        _rayOrigin,
        _down,
        Math.max(1.25, -this.velocity.y * dt + 1.05),
        this.carriedProp?.body,
        _groundHit
      );
      if (
        this.velocity.y <= 0 &&
        groundHit.hit &&
        groundHit.normal.y > 0.52 &&
        (!waterBasinAtXZ || groundHit.point.y >= waterBasinAtXZ.surfaceY - 0.05)
      ) {
        const targetY = groundHit.point.y + GROUND_FOOT_CLEARANCE;
        if (this.position.y <= targetY + 0.08) {
          const impactVy = Math.abs(this.velocity.y);
          this.position.y = targetY;
          this.velocity.y = 0;
          this.isHopAirborne = false;
          this.perchHopFloatTimer = 0;
          this.coyoteTimer = 0.12;
          this.hopPhase = 0;
          this.perchSettleDuration = 0.24;
          this.perchSettleTimer = 0.24;
          this.perchSettleAmp = Math.min(0.034, 0.016 + impactVy * 0.0024);
          this.audio.playHop(this.detectGroundSurfaceType(this.position));
        }
      } else if (
        this.velocity.y <= 0 &&
        waterBasinAtXZ &&
        this.position.y <= waterBasinAtXZ.surfaceY + 0.42
      ) {
        // Playful water-surface hop identical on both the river and fountain bowls!
        this.position.y = waterBasinAtXZ.surfaceY + 0.40;
        this.velocity.y = Math.max(4.6, -this.velocity.y * 0.45);
        this.isHopAirborne = true;
        this.multiSpaceWindow = 0.44;
        this.coyoteTimer = 0.12;
        this.hopPhase = 0.18;
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.24);
        this.flightStamina = Math.min(1.0, this.flightStamina + 0.22);
        _ripplePos.set(this.position.x, waterBasinAtXZ.surfaceY, this.position.z);
        this.fx.spawnWaterSplash(_ripplePos, 0.65);
        this.audio.playWaterPlop(this.position);
      } else if (this.velocity.y < -6.5 && !this.isInsideBirdcage(this.position)) {
        this.isHopAirborne = false;
        this.mode = 'flying';
      }
      this.applyPerimeterBreeze(dt);
      return;
    }

    if (this.perchIgnoreTimer <= 0 && moveDir.lengthSq() < 0.25) {
      const nearPerch = this.perches.findNearestPerch(
        this.position,
        0.88,
        this.lastPerchId,
        false,
        _perchQueryScratch
      );
      if (nearPerch && Math.abs(nearPerch.worldPoint.y - this.position.y) < 0.72) {
        this.enterPerch(nearPerch.segment, nearPerch.t, hSpeed);
        return;
      }
    }

    _rayOrigin.set(this.position.x, this.position.y + 0.92, this.position.z);
    const groundHit = this.physics.raycastGround(
      _rayOrigin,
      _down,
      3.5,
      this.carriedProp?.body,
      _groundHit
    );
    const solidAboveWater =
      groundHit.hit &&
      groundHit.normal.y > 0.52 &&
      (!waterBasinAtXZ || groundHit.point.y >= waterBasinAtXZ.surfaceY - 0.05);

    if (solidAboveWater) {
      const targetY = groundHit.point.y + GROUND_FOOT_CLEARANCE;
      if (targetY < this.position.y - 1.35) {
        if (hSpeed < 5.2 && this.tryAutoPerchOnNearbyLedge(1.15)) {
          return;
        }
        this.mode = 'flying';
        this.velocity.y = 0.2;
      } else if (targetY > this.position.y && targetY - this.position.y < 1.15) {
        // Snap immediately when feet would dip below the floor surface so claws never clip through floors/rugs
        this.position.y = targetY;
        this.velocity.y = 0;
      } else {
        this.position.y += (targetY - this.position.y) * Math.min(1, dt * 32);
        this.velocity.y = 0;
      }
    } else if (
      waterBasinAtXZ &&
      Math.abs(this.position.y - (waterBasinAtXZ.surfaceY + 0.35)) < 1.15
    ) {
      if (this.dizzyState === 'none') {
        this.position.y = waterBasinAtXZ.surfaceY + 0.40;
        this.velocity.y = 4.6;
        this.isHopAirborne = true;
        this.multiSpaceWindow = 0.44;
        this.hopPhase = 0.18;
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.24);
        _ripplePos.set(this.position.x, waterBasinAtXZ.surfaceY, this.position.z);
        this.fx.spawnWaterSplash(_ripplePos, 0.62);
        this.audio.playWaterPlop(this.position);
      } else {
        const buoyantY = waterBasinAtXZ.surfaceY + 0.35;
        this.position.y += (buoyantY - this.position.y) * Math.min(1, dt * 18);
        this.velocity.y = 0;
      }
    } else {
      if (hSpeed < 5.2 && this.tryAutoPerchOnNearbyLedge(1.15)) {
        return;
      }
      this.mode = 'flying';
    }

    this.applyPerimeterBreeze(dt);
  }

  private updateFlying(
    dt: number,
    cameraYaw: number,
    axes: {
      forward: number;
      right: number;
      braking: boolean;
      diving: boolean;
      flutterIntensity: number;
      mobileClimb: number;
    }
  ): void {
    this.hopPhase = 0;
    const weightPenalty = this.carriedProp ? this.carriedProp.beakWeight * 2.2 : 0;

    // 1. Measured Wingbeat Climb Burst (`Space` tap with 0.10s input buffer + 0.12s ledge coyote time)
    if (this.hasBufferedJump() && (this.flapCooldown <= 0 || this.coyoteTimer > 0)) {
      const usedCoyote = this.coyoteTimer > 0;
      this.consumeBufferedJump();
      this.coyoteTimer = 0;

      const ceilingFade = Math.max(0.18, Math.min(1.0, (158 - this.position.y) / 22));
      const staminaFactor = usedCoyote
        ? 1.0
        : 0.38 + 0.62 * Math.min(1.0, this.flightStamina / 0.26);
      const flapLift = (7.2 - weightPenalty * 0.55) * ceilingFade * staminaFactor;
      this.velocity.y = Math.min(11.8, Math.max(this.velocity.y * 0.42 + flapLift * 0.85, flapLift));

      if (!usedCoyote) {
        this.flightStamina = Math.max(0.0, this.flightStamina - 0.10);
      }
      this.boundCycleTimer = 0.0;
      this.wingTuckAmount = 0.0;

      const fwd = this.getForwardVector(_birdFwd);
      const speedH = Math.hypot(this.velocity.x, this.velocity.z);
      if (speedH < 16.5) {
        this.velocity.x += fwd.x * (3.4 * staminaFactor);
        this.velocity.z += fwd.z * (3.4 * staminaFactor);
      }

      this.flapAnimTimer = 0.85;
      this.flapCooldown = 0.16;
      this.flightWingbeatTimer = 0.12;
      this.audio.playWingFlap(staminaFactor);
      this.worldHandles?.applyAerodynamicWash?.(this.position, this.velocity, 0.72 * staminaFactor);
      this.physics.applyWindGustAt(this.position, this.velocity, 3.4, 0.72 * staminaFactor);
      if (this.flightStamina < 0.18) {
        this.fx.spawnBurst(this.position, 2, false, 1.4);
      }
    }

    // 2. Passer domesticus Neuromuscular Banking, Turning & Pitch Aerodynamics
    let turnInput = axes.diving ? 0 : axes.right;
    if (
      !axes.diving &&
      !this.isMobileInputActive() &&
      axes.forward > 0.1 &&
      Math.abs(axes.right) < 0.05
    ) {
      let camDiff = cameraYaw - this.yaw;
      while (camDiff > Math.PI) camDiff -= Math.PI * 2;
      while (camDiff < -Math.PI) camDiff += Math.PI * 2;
      turnInput = Math.max(-1, Math.min(1, -camDiff * 1.65));
    }

    if (axes.diving) {
      // Helicopter flutter-down: zero yaw rotation and level wings regardless of left/right drag angle
      this.bankAngle += (0 - this.bankAngle) * Math.min(1, dt * 18.0);
    } else {
      const turnRate = axes.braking ? 3.9 : 3.55;
      this.yaw -= turnInput * turnRate * dt;
      this.bankAngle += (turnInput - this.bankAngle) * Math.min(1, dt * 11.2);
    }

    const fwdX = Math.sin(this.yaw);
    const fwdZ = Math.cos(this.yaw);
    let hSpeed = Math.hypot(this.velocity.x, this.velocity.z);
    const isLongPressing =
      this.isLongPressFlying || (this.mobileControls?.isLongPressActive() ?? false);
    const holdingFlapCruise = isLongPressing && !axes.diving && this.flightStamina > 0.05;

    if (this.bonkTimer <= 0.35) {
      if (axes.braking) {
        this.wingTuckAmount = 0;
        hSpeed = Math.max(1.5, hSpeed * Math.pow(0.06, dt));
        if (this.velocity.y < -5.2) {
          this.velocity.y += (-5.2 - this.velocity.y) * Math.min(1, dt * 9.5);
        } else {
          this.velocity.y = Math.max(-5.2, this.velocity.y - 9.0 * dt);
        }
        this.pitchAngle += (-0.58 - this.pitchAngle) * Math.min(1, dt * 8);
      } else if (axes.diving) {
        // House Sparrow Helicopter Flutter-Down (drag down):
        // Strictly straight vertical line down (velocity.x = 0, velocity.z = 0), locked yaw (no rotation),
        // rapid horizontal-rotor wing flaps, and crisp vertical descent (-14.0 to -23.0 SU/s).
        const fi = Math.max(0.2, Math.min(1, axes.flutterIntensity));
        this.wingTuckAmount = 0.0;
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.52);
        this.flightStamina = Math.max(0.16, this.flightStamina - dt * 0.036);

        hSpeed = hSpeed < 0.14 ? 0 : hSpeed * Math.pow(0.00003, dt);

        const targetFlutterVy = -THREE.MathUtils.lerp(14.0, 23.0, fi) - weightPenalty * 0.65;
        this.velocity.y += (targetFlutterVy - this.velocity.y) * Math.min(1, dt * 16.0);

        const targetFlutterPitch = -0.28 - fi * 0.12;
        this.pitchAngle += (targetFlutterPitch - this.pitchAngle) * Math.min(1, dt * 14.5);
      } else if (holdingFlapCruise) {
        // Long-press powered upward flight:
        // Base climb angle = 30° (π/6 rad); dragging up increases climb angle up to 60° (π/3 rad)
        const upDrag = Math.max(0, Math.min(1, axes.mobileClimb));
        const climbAngleRad = THREE.MathUtils.degToRad(30 + upDrag * 30);

        const prevBound = this.boundCycleTimer;
        const flapCadence = 3.2 + upDrag * 1.4;
        this.boundCycleTimer = (this.boundCycleTimer + dt * flapCadence) % 1.0;
        if (this.boundCycleTimer < prevBound) {
          this.flightWingbeatTimer = 0.095;
          this.audio.playWingFlap(0.58 + upDrag * 0.28);
          this.worldHandles?.applyAerodynamicWash?.(this.position, this.velocity, 0.65);
          this.physics.applyWindGustAt(this.position, this.velocity, 3.2, 0.65);
        }

        this.flightStamina = Math.max(0.18, this.flightStamina - dt * 0.045);
        this.wingTuckAmount = 0.0;
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.38 + upDrag * 0.36);

        const ceilingFade = Math.max(0.05, Math.min(1.0, (158 - this.position.y) / 22));
        const totalClimbSpeed =
          (12.6 + upDrag * 1.5 - weightPenalty * 0.65) * (0.72 + 0.28 * this.flightStamina);
        const targetHSpeed = totalClimbSpeed * Math.cos(climbAngleRad);
        const targetVy = totalClimbSpeed * Math.sin(climbAngleRad) * ceilingFade;

        if (this.velocity.y < -4.0) {
          const nextVy = this.velocity.y + (targetVy - this.velocity.y) * Math.min(1, dt * 11.0);
          const shedDown = Math.max(0, nextVy - this.velocity.y);
          this.velocity.y = nextVy;
          hSpeed = Math.min(25.5, hSpeed + shedDown * 0.35);
        } else {
          hSpeed += (targetHSpeed - hSpeed) * Math.min(1, dt * 8.8);
          this.velocity.y += (targetVy - this.velocity.y) * Math.min(1, dt * 9.2);
        }

        const targetPitch = -climbAngleRad; // -30° (-0.5236 rad) to -60° (-1.0472 rad)
        this.pitchAngle += (targetPitch - this.pitchAngle) * Math.min(1, dt * 11.5);
      } else {
        // Paraglider Glide (~ -10° buoyant glide slope, L/D ≈ 5.67:1):
        // Maintains pressurized forward canopy airspeed (~11.8 m/s) instead of stalling into a steep drop
        this.wingTuckAmount = 0.0;
        const trimCruise = axes.forward > 0.1 ? 12.8 : 11.8;
        const dragRate = hSpeed > trimCruise + 0.8 ? 0.55 : 4.2;
        hSpeed += (trimCruise - hSpeed) * Math.min(1, dt * dragRate);

        const baseGlideDeg = 10.0 + weightPenalty * 1.35;
        const swoopBonusDeg = Math.min(4.0, Math.max(0, hSpeed - 13.2) * 0.42);
        const glideAngleRad = THREE.MathUtils.degToRad(baseGlideDeg - swoopBonusDeg);
        const targetGlideVy = -Math.max(6.5, hSpeed) * Math.tan(glideAngleRad);

        if (this.velocity.y < targetGlideVy - 0.6) {
          const nextVy = this.velocity.y + (targetGlideVy - this.velocity.y) * Math.min(1, dt * 8.5);
          const shedDown = Math.max(0, nextVy - this.velocity.y);
          this.velocity.y = nextVy;
          hSpeed = Math.min(22.0, hSpeed + shedDown * 0.36);
        } else if (this.velocity.y > targetGlideVy) {
          // Cresting gently from a 30°–60° climb into the -10° paraglider slope
          this.velocity.y = Math.max(targetGlideVy, this.velocity.y - (6.8 + weightPenalty * 0.7) * dt);
        } else {
          this.velocity.y += (targetGlideVy - this.velocity.y) * Math.min(1, dt * 7.5);
        }

        // Natural stamina recovery while gliding:
        this.flightStamina = Math.min(1.0, this.flightStamina + dt * 0.32);

        // Align body attitude along the -10° flight path with ~1.8° positive wing angle-of-attack (~ +8.2° pitch)
        const pathPitch = Math.atan2(-this.velocity.y, Math.max(4.5, hSpeed));
        const targetGlidePitch = Math.max(-0.45, Math.min(0.22, pathPitch - 0.032));
        this.pitchAngle += (targetGlidePitch - this.pitchAngle) * Math.min(1, dt * 8.5);
      }

      if (axes.diving) {
        if (hSpeed <= 1e-3) {
          this.velocity.x = 0;
          this.velocity.z = 0;
        } else {
          const prevH = Math.hypot(this.velocity.x, this.velocity.z);
          if (prevH > 1e-4) {
            this.velocity.x = (this.velocity.x / prevH) * hSpeed;
            this.velocity.z = (this.velocity.z / prevH) * hSpeed;
          } else {
            this.velocity.x = 0;
            this.velocity.z = 0;
          }
        }
      } else {
        const curDirX = hSpeed > 0.01 ? this.velocity.x / hSpeed : fwdX;
        const curDirZ = hSpeed > 0.01 ? this.velocity.z / hSpeed : fwdZ;
        const steerBlend = Math.min(1, dt * 10.4);
        _nextDir2.set(
          curDirX + (fwdX - curDirX) * steerBlend,
          curDirZ + (fwdZ - curDirZ) * steerBlend
        );
        if (_nextDir2.lengthSq() > 0.0001) _nextDir2.normalize();
        this.velocity.x = _nextDir2.x * hSpeed;
        this.velocity.z = _nextDir2.y * hSpeed;
      }
    } else {
      this.velocity.y -= 14.0 * dt;
    }

    // Wingbeat foley fires during active climb, rapid helicopter flutter-down, or airbraking — passive -10° paraglider glide stays serene
    this.flightWingbeatTimer -= dt;
    if (this.flightWingbeatTimer <= 0 && this.wingTuckAmount < 0.42) {
      if (axes.diving) {
        this.flightWingbeatTimer = 0.044;
        this.audio.playFlightWingbeat(0.38 + axes.flutterIntensity * 0.22);
      } else if (this.flapAnimTimer > 0.02) {
        this.flightWingbeatTimer = 0.076;
        this.audio.playFlightWingbeat(0.42 + 0.28 * Math.min(1.0, this.flapAnimTimer / 0.82));
      } else if (axes.braking) {
        this.flightWingbeatTimer = 0.088;
        this.audio.playFlightWingbeat(0.46);
      }
    }

    // 3. Thermal Updraft Lift (skipped during intentional helicopter flutter-down so vertical descent stays crisp)
    if (!axes.diving) {
      const thermalLift = this.physics.sampleUpdraftLift(this.position);
      if (thermalLift > 0) {
        this.wingTuckAmount = 0.0;
        this.flightStamina = Math.min(1.0, this.flightStamina + dt * 0.95);
        this.velocity.y = Math.min(16.0, this.velocity.y + thermalLift * dt);
      }
    }

    // 4. Perimeter Wind Return
    this.applyPerimeterBreeze(dt);

    // 5. Single-Ray Velocity Sweep, Landing & Dizzy Wall-Hit / Cute Bump
    _moveStep.copy(this.velocity).multiplyScalar(dt);
    const stepLen = _moveStep.length();
    if (stepLen > 0.001) {
      _sweepDir.copy(_moveStep).divideScalar(stepLen);
      const hit = this.physics.raycastGround(
        this.position,
        _sweepDir,
        stepLen + 0.58,
        this.carriedProp?.body,
        _flySweepHit
      );
      if (hit.hit) {
        const hitBasin = this.physics.getBasinAtXZ(hit.point.x, hit.point.z, hit.point.y);
        const submergedFloor = hitBasin !== null && hit.point.y < hitBasin.surfaceY - 0.05;

        if (!submergedFloor && hit.normal.y > 0.55 && this.velocity.y <= 1.8) {
          if (this.checkTrampolineBounce(hit.point)) {
            return;
          }
          const impactSpeed = this.velocity.length();

          if (this.tryAutoPerchOnNearbyLedge(1.38 + stepLen)) {
            return;
          }

          this.position.set(hit.point.x, hit.point.y + GROUND_FOOT_CLEARANCE, hit.point.z);
          this.velocity.y = 0;
          this.isHopAirborne = false;
          this.coyoteTimer = 0.12;
          this.mode = 'grounded';
          this.wingTuckAmount = 0;
          this.bankAngle = 0;
          this.pitchAngle = 0;
          this.perchSettleDuration = 0.20;
          this.perchSettleTimer = 0.20;
          this.perchSettleAmp = Math.min(0.034, 0.014 + impactSpeed * 0.0018);
          const surf = this.detectGroundSurfaceType(this.position);
          this.audio.playLanding(impactSpeed, surf);
          this.fx.spawnBurst(hit.point, impactSpeed > 10 ? 8 : 5, false, impactSpeed > 10 ? 3.2 : 2.2);
          return;
        } else if (!submergedFloor) {
          if (hit.distance < 0.52) {
            this.position.addScaledVector(hit.normal, 0.52 - hit.distance);
          }

          if (this.velocity.y < 3.6 && this.tryAutoPerchOnNearbyLedge(1.8)) {
            return;
          }

          const vn = this.velocity.dot(hit.normal);
          if (vn < 0) {
            const impactIntoWall = -vn;
            if (this.tutorialSafeZone) {
              this.velocity.addScaledVector(hit.normal, -vn * 1.05);
            } else {
              const meta = this.physics.getBumpMetadata(hit.body);
              const isSoftOrInteractive = meta?.causesDizzy === false || (hit.body && hit.body.mass > 0);
              if (!isSoftOrInteractive && Math.abs(hit.normal.y) < 0.38 && impactIntoWall > 6.5) {
                this.triggerDizzyWallFall(hit.point, hit.normal, hit.body, impactIntoWall);
                return;
              } else if (impactIntoWall > 4.6 && this.bonkCooldown <= 0) {
                this.triggerCuteBump(hit.point, hit.normal, hit.body, impactIntoWall);
              } else {
                this.velocity.addScaledVector(hit.normal, -vn * 1.15);
              }
            }
          }
        }
      }
    }

    // 5B. Wide-Angle Static Wall Whiskers + Ceiling + 2nd-Pass Corner Clearance
    // Prevents shallow grazing-angle flight, prop-masked walls, and 90° corners from tunneling
    const flyHSpeed = Math.hypot(this.velocity.x, this.velocity.z);
    if (flyHSpeed > 0.04) {
      _hDir.set(this.velocity.x / flyHSpeed, 0, this.velocity.z / flyHSpeed);
      const whiskerReach = 0.58 + flyHSpeed * dt;
      for (let i = 0; i < FLY_WALL_WHISKERS.length; i++) {
        const ang = FLY_WALL_WHISKERS[i];
        const cosA = Math.cos(ang);
        const sinA = Math.sin(ang);
        _rayDir.set(_hDir.x * cosA - _hDir.z * sinA, 0, _hDir.x * sinA + _hDir.z * cosA);
        const wHit = this.physics.raycastStaticCameraObstacle(this.position, _rayDir, whiskerReach);
        if (wHit.hit && Math.abs(wHit.normal.y) < 0.48) {
          if (wHit.distance < 0.55) {
            this.position.addScaledVector(wHit.normal, 0.55 - wHit.distance);
          }
          const vn = this.velocity.dot(wHit.normal);
          if (vn < 0) {
            this.velocity.addScaledVector(wHit.normal, -vn * 1.03);
          }
        }
      }
    }

    if (this.velocity.y > 0.05) {
      const ceilHit = this.physics.raycastStaticCameraObstacle(
        this.position,
        _upDir,
        0.56 + this.velocity.y * dt
      );
      if (ceilHit.hit && ceilHit.normal.y < -0.45) {
        if (ceilHit.distance < 0.54) {
          this.position.addScaledVector(ceilHit.normal, 0.54 - ceilHit.distance);
        }
        const vn = this.velocity.dot(ceilHit.normal);
        if (vn < 0) {
          this.velocity.addScaledVector(ceilHit.normal, -vn * 1.03);
        }
      }
    }

    _moveStep.copy(this.velocity).multiplyScalar(dt);
    const postStepLen = _moveStep.length();
    if (postStepLen > 0.001) {
      _sweepDir.copy(_moveStep).divideScalar(postStepLen);
      const cornerHit = this.physics.raycastStaticCameraObstacle(
        this.position,
        _sweepDir,
        postStepLen + 0.56
      );
      if (cornerHit.hit && cornerHit.normal.y < 0.52) {
        if (cornerHit.distance < 0.54) {
          this.position.addScaledVector(cornerHit.normal, 0.54 - cornerHit.distance);
        }
        const vn = this.velocity.dot(cornerHit.normal);
        if (vn < 0) {
          this.velocity.addScaledVector(cornerHit.normal, -vn * 1.02);
        }
      }
    }

    this.position.addScaledVector(this.velocity, dt);
    if (this.position.y > 162.0) {
      this.position.y = 162.0;
      if (this.velocity.y > 0) this.velocity.y = 0;
    }

    if (this.velocity.y < 1.0 && this.checkTrampolineBounce(this.position)) {
      return;
    }

    const basinAtXZ = this.physics.getBasinAtXZ(
      this.position.x,
      this.position.z,
      this.position.y
    );
    if (this.velocity.y < 5.2 && this.perchIgnoreTimer <= 0) {
      let searchR = axes.diving ? 0.88 : axes.braking ? 2.15 : hSpeed < 11.5 ? 1.72 : 1.35;
      if (basinAtXZ && this.position.y < basinAtXZ.surfaceY + 1.25 && !axes.braking) {
        const toCenterX = basinAtXZ.center.x - this.position.x;
        const toCenterZ = basinAtXZ.center.z - this.position.z;
        if (this.velocity.x * toCenterX + this.velocity.z * toCenterZ > 0) {
          searchR = 0.65;
        }
      }
      if (this.tryAutoPerchOnNearbyLedge(searchR)) {
        return;
      }
    }

    const waterBasin = this.physics.isPointInBasin(
      this.position.x,
      this.position.y - 0.35,
      this.position.z,
      0.55
    );
    if (waterBasin && this.velocity.y < 0) {
      this.position.y = waterBasin.surfaceY + 0.42;
      this.velocity.y = Math.max(4.5, -this.velocity.y * 0.45);
      this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.26);
      this.flightStamina = Math.min(1.0, this.flightStamina + 0.25);
      _ripplePos.set(this.position.x, waterBasin.surfaceY, this.position.z);
      this.fx.spawnWaterSplash(_ripplePos, 0.72);
      this.audio.playWaterPlop(this.position);
      return;
    }

    // Continuous downward foot-clearance probe during shallow floor-skimming flight
    if (!basinAtXZ || this.position.y > basinAtXZ.surfaceY + 0.65) {
      _rayOrigin.set(this.position.x, this.position.y + 0.56, this.position.z);
      const skimFloorHit = this.physics.raycastGround(
        _rayOrigin,
        _down,
        0.56 + GROUND_FOOT_CLEARANCE + 0.05,
        this.carriedProp?.body,
        _groundHit
      );
      if (
        skimFloorHit.hit &&
        skimFloorHit.normal.y > 0.55 &&
        (!basinAtXZ || skimFloorHit.point.y >= basinAtXZ.surfaceY - 0.05) &&
        this.position.y < skimFloorHit.point.y + GROUND_FOOT_CLEARANCE
      ) {
        this.position.y = skimFloorHit.point.y + GROUND_FOOT_CLEARANCE;
        if (this.velocity.y <= 1.2) {
          const impactSpeed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
          this.velocity.y = 0;
          this.isHopAirborne = false;
          this.coyoteTimer = 0.12;
          this.mode = 'grounded';
          const surf = this.detectGroundSurfaceType(this.position);
          this.audio.playLanding(impactSpeed, surf);
          this.fx.spawnBurst(skimFloorHit.point, 4, false, 1.8);
          this.applyPerimeterBreeze(dt);
          return;
        }
      }
    }

    if (basinAtXZ) {
      if (this.position.y < basinAtXZ.surfaceY + 0.35) {
        this.position.y = basinAtXZ.surfaceY + 0.42;
        this.velocity.y = Math.max(4.5, -this.velocity.y * 0.45);
        this.flapAnimTimer = Math.max(this.flapAnimTimer, 0.24);
        _ripplePos.set(this.position.x, basinAtXZ.surfaceY, this.position.z);
        this.fx.spawnWaterSplash(_ripplePos, 0.65);
        this.audio.playWaterPlop(this.position);
      }
    } else if (this.position.y < -0.25) {
      this.position.y = GROUND_FOOT_CLEARANCE;
      this.velocity.y = 0;
      this.isHopAirborne = false;
      this.coyoteTimer = 0.12;
      this.mode = 'grounded';
    }

    this.applyPerimeterBreeze(dt);
  }

  private checkTrampolineBounce(pos: THREE.Vector3): boolean {
    const trampolines = this.physics.trampolines;
    for (let i = 0; i < trampolines.length; i++) {
      const tz = trampolines[i];
      const dx = Math.abs(pos.x - tz.center.x);
      const dy = Math.abs(pos.y - tz.center.y);
      const dz = Math.abs(pos.z - tz.center.z);
      if (dx <= tz.halfExtents.x && dy <= tz.halfExtents.y + 0.55 && dz <= tz.halfExtents.z) {
        this.position.y = tz.center.y + tz.halfExtents.y + 0.56;
        this.velocity.y = tz.bounceSpeed;
        this.flightStamina = 1.0;
        this.isHopAirborne = false;
        this.coyoteTimer = 0;
        this.mode = 'flying';
        this.wingTuckAmount = 0;
        this.flapAnimTimer = 0.65;
        this.physics.triggerTrampolineSquash(tz);
        this.audio.playAwningBounce();
        _fxPos.set(pos.x, tz.center.y + tz.halfExtents.y, pos.z);
        this.fx.spawnBurst(_fxPos, 7, false, 4.2);
        return true;
      }
    }
    return false;
  }

  private checkFootSplash(): boolean {
    const basin = this.physics.isPointInBasin(
      this.position.x,
      this.position.y - GROUND_FOOT_CLEARANCE,
      this.position.z,
      0.48
    );
    if (basin) {
      _ripplePos.set(this.position.x, basin.surfaceY, this.position.z);
      this.fx.spawnWaterRipple(_ripplePos, 1.6);
      return true;
    }
    return false;
  }

  private applyPerimeterBreeze(_dt: number): void {
    // Physical perimeter buildings, corner palazzos, and boulevard galleria gatehouses at ±148..166
    // block flight naturally; keep only a far backstop clamp strictly behind those solid facades.
    const hardBackstop = 168.0;
    if (Math.abs(this.position.x) > hardBackstop) {
      this.position.x = Math.sign(this.position.x) * hardBackstop;
      if (this.velocity.x * this.position.x > 0) this.velocity.x = 0;
    }
    if (Math.abs(this.position.z) > hardBackstop) {
      this.position.z = Math.sign(this.position.z) * hardBackstop;
      if (this.velocity.z * this.position.z > 0) this.velocity.z = 0;
    }
  }

  public triggerIntroChirp(): void {
    this.chirpOpen = 1.0;
    this.introChirpTimer = 0.56;
    this.hopPhase = 0.24;
    this.flapAnimTimer = 0.22;
    this.audio.playSparrowChirp(true);
  }

  public triggerChirp(): void {
    this.chirpCount++;
    const trill = this.mode === 'flying' || Math.random() > 0.55;
    this.chirpOpen = 1.0;
    this.audio.playSparrowChirp(trill);

    this.sparrowMesh.root.updateMatrixWorld(true);
    this.sparrowMesh.beakHoldAnchor.getWorldPosition(_chirpEmitPos);
    this.fx.spawnChirpRing(_chirpEmitPos, PALETTE.goldGlow, this.mode === 'flying' ? 9.5 : 7.0);
    this.onChirpCallback?.(_chirpEmitPos);
  }

  public triggerMischiefChirp(): void {
    this.chirpOpen = Math.max(this.chirpOpen, 0.85);
    this.audio.playSparrowMicroChirp('mischief');
  }

  private checkAutoSeedNibbles(): void {
    const props = this.physics.props;
    for (let i = 0; i < props.length; i++) {
      const p = props[i];
      if (p.consumed || p.waterRespawnTimer !== undefined || p.kind !== 'seed') continue;
      if (p.mesh.position.distanceToSquared(this.position) < 1.12 * 1.12) {
        this.consumeSeed(p);
      }
    }
  }

  private updateMobileAutoInteract(dt: number): void {
    if (this.mobileDropIgnoreGrabTimer > 0) {
      this.mobileDropIgnoreGrabTimer = Math.max(0, this.mobileDropIgnoreGrabTimer - dt);
    }
    if (this.mobileWindowPushCooldown > 0) {
      this.mobileWindowPushCooldown = Math.max(0, this.mobileWindowPushCooldown - dt);
    }
    if (this.mobileSwitchCooldowns.size > 0) {
      for (const [id, rem] of this.mobileSwitchCooldowns) {
        const next = rem - dt;
        if (next <= 0) {
          this.mobileSwitchCooldowns.delete(id);
        } else {
          this.mobileSwitchCooldowns.set(id, next);
        }
      }
    }

    const isGestureActive =
      this.isMobileInputActive() ||
      this.isLongPressFlying ||
      (this.mobileControls?.stick.active ?? false);
    if (
      !this.inputEnabled ||
      !isGestureActive ||
      this.dizzyState !== 'none' ||
      this.introLookTarget !== null
    ) {
      return;
    }

    // 1. Proximity auto-tug on Atelier window pull-cord ring
    if (
      this.worldHandles &&
      !this.worldHandles.isWindowBarUnlatched() &&
      this.position.distanceTo(this.worldHandles.windowPullRingPos) <= 2.45
    ) {
      this.peckTimer = Math.max(this.peckTimer, 0.14);
      this.worldHandles.tugWindowPullCord(dt * 8.2);
    }

    // 2. Proximity auto-push on ajar French casement window sashes once unlatched
    if (
      this.worldHandles &&
      this.worldHandles.isWindowBarUnlatched() &&
      !this.worldHandles.isWindowUnlatched() &&
      this.mobileWindowPushCooldown <= 0
    ) {
      if (this.worldHandles.pushWindowSashes(this.position, this.velocity, true)) {
        this.peckTimer = 0.20;
        this.mobileWindowPushCooldown = 0.28;
        this.audio.playPeckTap('wood', this.position);
      }
    }

    // 3. Proximity auto-peck switches (cage latch, picnic basket, bells, shutters, levers, etc.)
    const switches = this.physics.peckSwitches;
    const fwd = this.getForwardVector(_actionFwd);
    _beakReachPos.copy(this.position).addScaledVector(fwd, 0.55);
    for (let i = 0; i < switches.length; i++) {
      const sw = switches[i];
      if (sw.cooldown > 0 || (this.mobileSwitchCooldowns.get(sw.id) ?? 0) > 0) continue;
      if (sw.id === 'indoor_cage_latch' && this.worldHandles?.isCageUnlatched()) continue;
      if (sw.id === 'attic_window_pull_ring' && this.worldHandles?.isWindowBarUnlatched()) continue;

      const distToBird = sw.position.distanceTo(this.position);
      const distToBeak = sw.position.distanceTo(_beakReachPos);
      const extraRadius =
        sw.id === 'indoor_cage_latch' && this.mode === 'flying' ? 0.65 : 0.22;
      if (Math.min(distToBird, distToBeak) <= sw.radius + extraRadius) {
        sw.cooldown = 0.10;
        this.peckTimer = 0.22;
        this.chirpOpen = Math.max(this.chirpOpen, 0.55);
        this.audio.playBeakSnap(true);
        this.fx.spawnBurst(sw.position, 5, true, 2.5);
        sw.onPeck();

        const repeatDelay =
          sw.id === 'indoor_cage_latch'
            ? 0.36
            : sw.id === 'attic_window_pull_ring'
              ? 0.45
              : sw.id === 'gazebo_picnic_basket' ||
                  sw.id === 'bookshop_swinging_sign' ||
                  sw.id === 'postcard_carousel_spin'
                ? 2.2
                : 4.5;
        this.mobileSwitchCooldowns.set(sw.id, repeatDelay);
        return;
      }
    }

    // 4. Proximity auto-grab grabbable props when beak is empty
    if (!this.carriedProp && this.mobileDropIgnoreGrabTimer <= 0) {
      let bestProp: InteractiveProp | null = null;
      let bestDist = 1.65;
      const props = this.physics.props;
      for (let i = 0; i < props.length; i++) {
        const p = props[i];
        if (p.consumed || p.waterRespawnTimer !== undefined || !p.grabbable || p.kind === 'seed') {
          continue;
        }
        const dist = p.mesh.position.distanceTo(this.position);
        if (dist < bestDist) {
          bestDist = dist;
          bestProp = p;
        }
      }
      if (bestProp) {
        bestProp.carried = true;
        bestProp.body.sleep();
        this.carriedProp = bestProp;
        this.peckTimer = 0.20;
        this.chirpOpen = Math.max(this.chirpOpen, 0.55);
        this.audio.playPeckTap('wood', bestProp.mesh.position);
        this.audio.playBeakSnap(true);
        this.audio.addChaosImpulse(0.24);
        if (Math.random() < 0.55) {
          this.audio.playSparrowMicroChirp('mischief');
        }
        this.fx.spawnBurst(bestProp.mesh.position, 4, true, 2.0);
        this.onPropPickedUpCallback?.(bestProp);
      }
    }
  }

  private consumeSeed(p: InteractiveProp): void {
    _seedConsumePos.copy(p.mesh.position);
    this.physics.removeProp(p);
    this.seedsGathered++;
    if (this.carriedProp !== null) {
      // Gentle nibble while holding an item so carried baguettes/keys never dip into the floor
      this.eatTimer = 0.22;
      this.chirpOpen = Math.max(this.chirpOpen, 0.35);
    } else {
      this.eatTimer = 0.38;
      this.peckTimer = Math.max(this.peckTimer, 0.20);
      this.chirpOpen = Math.max(this.chirpOpen, 0.65);
    }
    this.audio.playBreadcrumbCrunch();
    if (Math.random() < 0.52) {
      this.audio.playSparrowMicroChirp('hop');
    }
    this.fx.spawnBreadcrumbCrumbs(_seedConsumePos, 9);
    this.onSeedEatenCallback?.(_seedConsumePos, this.seedsGathered);
  }

  public triggerPeck(): void {
    this.handleBeakAction();
  }

  private handleBeakAction(): void {
    this.peckTimer = 0.22;
    this.chirpOpen = Math.max(this.chirpOpen, 0.55);

    const fwd = this.getForwardVector(_actionFwd);
    _beakReachPos.copy(this.position).addScaledVector(fwd, 0.62);

    // 0. If carrying an item, raycast static obstacles first so dropped quest items never spawn inside/behind walls or cage bars
    if (this.carriedProp) {
      const prop = this.carriedProp;
      this.carriedProp = null;
      prop.carried = false;
      prop.body.wakeUp();

      _dropPos.copy(this.position).addScaledVector(fwd, 0.75);
      _dropPos.y += this.mode === 'flying' ? -0.35 : 0.25;

      let dropVx = this.velocity.x * 0.8 + fwd.x * 3.8;
      let dropVy = Math.max(-2.0, this.velocity.y * 0.5 + 2.4);
      let dropVz = this.velocity.z * 0.8 + fwd.z * 3.8;

      // Gentle magnetic drop-aim when releasing a quest item right beside its vignette target
      if (this.worldHandles) {
        const deliveryTarget =
          prop.kind === 'sugar_cube'
            ? this.worldHandles.sleepyMugPos
            : prop.kind === 'shiny_coin'
              ? this.worldHandles.guitarCasePos
              : prop.kind === 'brass_key'
                ? this.worldHandles.musicBoxPos
                : prop.kind === 'eyeglasses' || prop.kind === 'berry' || prop.kind === 'flower_sprig'
                  ? this.worldHandles.painterEaselPos
                  : prop.isNestMaterial
                    ? this.worldHandles.skySanctuaryNestPos
                    : undefined;
        if (deliveryTarget && this.position.distanceTo(deliveryTarget) < 2.9) {
          _dropPos.copy(this.position).lerp(deliveryTarget, 0.55);
          _dropPos.y = Math.max(_dropPos.y, deliveryTarget.y + 0.32);
          dropVx = (deliveryTarget.x - _dropPos.x) * 2.2;
          dropVy = -0.6;
          dropVz = (deliveryTarget.z - _dropPos.z) * 2.2;
        }
      }

      _dropDir.subVectors(_dropPos, this.position);
      const dropDist = _dropDir.length();
      if (dropDist > 1e-4) {
        _dropDir.divideScalar(dropDist);
        const _dropObstacleHit: StaticCameraHit = this.physics.raycastStaticCameraObstacle(
          this.position,
          _dropDir,
          dropDist + 0.26
        );
        if (_dropObstacleHit.hit) {
          _dropHitPt.copy(this.position).addScaledVector(_dropDir, _dropObstacleHit.distance);
          _dropPos.copy(_dropHitPt).addScaledVector(_dropObstacleHit.normal, 0.26);
          _dropDelta.subVectors(_dropPos, this.position);
          const clampedLen = _dropDelta.length();
          if (clampedLen > dropDist && clampedLen > 1e-4) {
            _dropPos.copy(this.position).addScaledVector(_dropDelta, dropDist / clampedLen);
          }
          const vn =
            dropVx * _dropObstacleHit.normal.x +
            dropVy * _dropObstacleHit.normal.y +
            dropVz * _dropObstacleHit.normal.z;
          if (vn < 0) {
            dropVx -= _dropObstacleHit.normal.x * vn * 1.15;
            dropVy -= _dropObstacleHit.normal.y * vn * 1.15;
            dropVz -= _dropObstacleHit.normal.z * vn * 1.15;
          }
        }
      }

      if (this.cageBounds?.locked) {
        const cb = this.cageBounds;
        const cdx = _dropPos.x - cb.center.x;
        const cdz = _dropPos.z - cb.center.z;
        const cr = Math.hypot(cdx, cdz);
        const maxDropR = Math.max(0.6, cb.radius - 0.26);
        if (cr > maxDropR && cr > 1e-4) {
          _dropPos.x = cb.center.x + (cdx / cr) * maxDropR;
          _dropPos.z = cb.center.z + (cdz / cr) * maxDropR;
        }
        _dropPos.y = Math.max(cb.floorY + 0.12, Math.min(cb.ceilingY - 0.35, _dropPos.y));
      }

      if (isInsideAtelier(this.position)) {
        _dropPos.x = Math.max(ATELIER_INNER_X_MIN + 0.18, _dropPos.x);
        _dropPos.z = Math.max(
          ATELIER_INNER_Z_MIN + 0.18,
          Math.min(ATELIER_INNER_Z_MAX - 0.18, _dropPos.z)
        );
        _dropPos.y = Math.max(26.42, Math.min(ATELIER_INNER_Y_MAX - 0.24, _dropPos.y));
        const inOpenAperture =
          !this.windowBarrier?.closed &&
          _dropPos.z >= ATELIER_APERTURE_Z_MIN + 0.15 &&
          _dropPos.z <= ATELIER_APERTURE_Z_MAX - 0.15 &&
          _dropPos.y >= ATELIER_APERTURE_Y_MIN + 0.12 &&
          _dropPos.y <= ATELIER_APERTURE_Y_MAX - 0.15;
        if (!inOpenAperture && _dropPos.x > ATELIER_INNER_X_MAX - 0.22) {
          _dropPos.x = ATELIER_INNER_X_MAX - 0.22;
          if (dropVx > 0) dropVx = 0;
        }
      }

      prop.body.position.set(_dropPos.x, _dropPos.y, _dropPos.z);
      prop.body.interpolatedPosition.set(_dropPos.x, _dropPos.y, _dropPos.z);
      prop.mesh.position.copy(_dropPos);
      prop.body.velocity.set(dropVx, dropVy, dropVz);
      this.audio.playBeakSnap(false);
      this.onPropDroppedCallback?.(prop);
      return;
    }

    // 1. Check close-range peckable switches
    const switches = this.physics.peckSwitches;
    for (let i = 0; i < switches.length; i++) {
      const sw = switches[i];
      const distToBird = sw.position.distanceTo(this.position);
      const distToBeak = sw.position.distanceTo(_beakReachPos);
      if (sw.cooldown <= 0 && Math.min(distToBird, distToBeak) <= sw.radius) {
        sw.cooldown = 0.09;
        this.audio.playBeakSnap(true);
        this.fx.spawnBurst(sw.position, 5, true, 2.5);
        sw.onPeck();
        return;
      }
    }

    // 2. Check close-range grabbable physics prop in front of beak (`<= 1.78 SU`)
    let bestProp: InteractiveProp | null = null;
    let bestScore = 1.82;
    const props = this.physics.props;
    for (let i = 0; i < props.length; i++) {
      const p = props[i];
      if (p.consumed || p.waterRespawnTimer !== undefined || !p.grabbable) continue;
      _propDelta.subVectors(p.mesh.position, this.position);
      const dist = _propDelta.length();
      if (dist > 1.78) continue;
      const align = dist > 0.05 ? _propDelta.dot(fwd) / dist : 1.0;
      if (align < -0.25 && dist > 0.95) continue;
      const score = dist - align * 0.35;
      if (score < bestScore) {
        bestScore = score;
        bestProp = p;
      }
    }

    if (bestProp) {
      if (bestProp.kind === 'seed') {
        this.consumeSeed(bestProp);
        return;
      }
      bestProp.carried = true;
      bestProp.body.sleep();
      this.carriedProp = bestProp;
      this.audio.playPeckTap('wood', bestProp.mesh.position);
      this.audio.playBeakSnap(true);
      this.audio.addChaosImpulse(0.24);
      if (Math.random() < 0.55) {
        this.audio.playSparrowMicroChirp('mischief');
      }
      this.fx.spawnBurst(bestProp.mesh.position, 4, true, 2.0);
      this.onPropPickedUpCallback?.(bestProp);
      return;
    }

    // 3. Check pecking Citizens, Park Pigeons, or Companion Songbirds
    if (this.npcs) {
      const npcHit = this.npcs.handlePlayerPeck(_beakReachPos, this.position);
      if (npcHit.hit) {
        this.audio.playPeckTap('fabric', _beakReachPos);
        this.audio.addChaosImpulse(0.32);
        this.triggerMischiefChirp();
        if (npcHit.shooImpulse) {
          this.applyShooImpulse(npcHit.shooImpulse);
        }
        return;
      }
    }

    // 4. Check pecking open the ajar French Casement Window Sashes
    if (this.worldHandles?.pushWindowSashes(this.position, this.velocity, true)) {
      this.audio.playPeckTap('wood', _beakReachPos);
      return;
    }

    // 5. Strike / crack / shatter dynamic props
    if (this.mode !== 'flying') {
      this.npcs?.notifyTablePeck?.(_beakReachPos);
    }
    const struck = this.physics.strikePropsWithPeck(_beakReachPos, fwd, 1.55);
    if (struck) {
      this.audio.addChaosImpulse(0.45);
      if (!struck.shattered) {
        const isCer =
          struck.prop.kind === 'ceramic_mug' ||
          struck.prop.kind === 'teacup' ||
          struck.prop.kind === 'porcelain_plate' ||
          struck.prop.kind === 'pigment_jar' ||
          struck.prop.kind === 'terracotta_pot';
        this.audio.playPeckTap(isCer ? 'ceramic' : 'wood', struck.prop.mesh.position);
        if (Math.random() < 0.5) {
          this.triggerMischiefChirp();
        }
        this.fx.spawnBurst(_beakReachPos, 4, false, 2.2);
      }
      return;
    }

    const nearWindowPullRing =
      Boolean(this.worldHandles) &&
      !this.worldHandles!.isWindowBarUnlatched() &&
      this.position.distanceTo(this.worldHandles!.windowPullRingPos) <= 2.35;

    if (
      this.mode === 'flying' &&
      !this.cageBounds?.locked &&
      !nearWindowPullRing &&
      this.droppingCooldown <= 0
    ) {
      this.peckTimer = 0;
      this.chirpOpen = 0;
      this.droppingCooldown = 0.88;
      this.droppingTailFlick = 0.2;
      this.spawnMidFlightDropping(fwd);
      return;
    }

    // 6. Tactile surface / air peck tap
    const surf = this.mode === 'flying' ? 'wood' : this.detectGroundSurfaceType(this.position);
    this.audio.playPeckTap(surf === 'stone' ? 'stone' : 'wood');
    this.fx.spawnBurst(_beakReachPos, 3, false, 1.5);
  }

  private spawnMidFlightDropping(fwd: THREE.Vector3): void {
    this.sparrowMesh.root.updateMatrixWorld(true);
    this.sparrowMesh.tailPivot.getWorldPosition(_droppingSpawnPos);
    _droppingSpawnPos.y -= 0.055;

    let slot: ActiveDropping | null = null;
    const len = this.activeDroppings.length;
    for (let step = 0; step < len; step++) {
      const idx = (this.droppingCursor + step) % len;
      if (!this.activeDroppings[idx].active) {
        slot = this.activeDroppings[idx];
        this.droppingCursor = (idx + 1) % len;
        break;
      }
    }
    if (!slot) {
      slot = this.activeDroppings[this.droppingCursor];
      this.droppingCursor = (this.droppingCursor + 1) % len;
    }

    slot.pos.copy(_droppingSpawnPos);
    slot.prevPos.copy(_droppingSpawnPos);
    slot.vel.set(
      this.velocity.x * 0.72 - fwd.x * 0.65,
      Math.min(1.2, this.velocity.y * 0.45) - 2.6,
      this.velocity.z * 0.72 - fwd.z * 0.65
    );
    slot.age = 0;
    slot.active = true;
    slot.mesh.position.copy(slot.pos);
    slot.mesh.scale.setScalar(1.0);
    _droppingTrailDir.copy(slot.vel);
    if (_droppingTrailDir.lengthSq() > 1e-4) {
      _droppingTrailDir.normalize().negate();
      slot.mesh.quaternion.setFromUnitVectors(_droppingUpAxis, _droppingTrailDir);
    }
    slot.mesh.visible = true;
    this.audio.playDroppingRelease();
  }

  /**
   * Spawns sculpted 3D crescent wingbeat airflow wisps & helicopter ground-cushion downwash
   * strictly during active wing-flutter (powered climb, airborne hop-flutter, braking flare,
   * and helicopter flutter-down). Never spawns anything during passive -10° paraglider glide!
   */
  private updateWingFlutterFX(
    dt: number,
    isHelicopter: boolean,
    flutterIntensity: number,
    isBraking: boolean
  ): void {
    if (this.dizzyState !== 'none') {
      this.wingFlutterFxTimer = 0;
      this.helicopterDownwashTimer = 0;
      return;
    }

    const activeFlyingFlutter =
      this.mode === 'flying' &&
      this.wingTuckAmount < 0.38 &&
      (isHelicopter || this.flapAnimTimer > 0.04 || isBraking);
    const activeHopFlutter =
      this.isHopAirborne && (this.flapAnimTimer > 0.02 || this.flyHoldPrimed);

    if (!activeFlyingFlutter && !activeHopFlutter) {
      // Passive paraglider glide or resting: strictly zero wingtip lines / FX
      this.wingFlutterFxTimer = 0;
      this.helicopterDownwashTimer = 0;
      return;
    }

    const interval = isHelicopter ? 0.025 : activeHopFlutter ? 0.042 : 0.035;
    this.wingFlutterFxTimer -= dt;
    if (this.wingFlutterFxTimer <= 0) {
      this.wingFlutterFxTimer = interval;
      this.sparrowMesh.wingTipLeft.getWorldPosition(_tipL);
      this.sparrowMesh.wingTipRight.getWorldPosition(_tipR);
      this.sparrowMesh.bodyPivot.getWorldQuaternion(_holdQuat);

      const intensity = isHelicopter
        ? 0.92 + flutterIntensity * 0.42
        : isBraking
          ? 0.95
          : activeHopFlutter
            ? 0.68
            : 0.76 + Math.min(0.36, this.flapAnimTimer * 0.45);

      this.fx.spawnWingFlutterArc(_tipL, _holdQuat, -1, isHelicopter, intensity);
      this.fx.spawnWingFlutterArc(_tipR, _holdQuat, 1, isHelicopter, intensity);

      if (isHelicopter && Math.random() < 0.14) {
        this.fx.spawnFeatherBurst(this.position, 1, 1.65);
      }
    }

    if (isHelicopter) {
      this.physics.applyWindGustAt(this.position, _down, 3.2, 0.48 * dt * 10.0);
      this.helicopterDownwashTimer -= dt;
      if (this.helicopterDownwashTimer <= 0) {
        this.helicopterDownwashTimer = 0.085;
        const groundCheck = this.physics.raycastGround(
          this.position,
          _down,
          4.4,
          this.carriedProp?.body,
          _groundHit
        );
        if (groundCheck.hit && groundCheck.normal.y > 0.5) {
          const prox = 1.0 - groundCheck.distance / 4.4;
          this.fx.spawnHelicopterDownwash(
            groundCheck.point,
            (0.62 + flutterIntensity * 0.48) * (0.55 + prox * 0.65)
          );
        }
      }
    } else {
      this.helicopterDownwashTimer = 0;
    }
  }

  private updateActiveDroppings(dt: number): void {
    for (let i = 0; i < this.activeDroppings.length; i++) {
      const slot = this.activeDroppings[i];
      if (!slot.active) continue;

      slot.age += dt;
      if (slot.age > 6.5 || slot.pos.y < -16.0) {
        slot.active = false;
        slot.mesh.visible = false;
        continue;
      }

      slot.prevPos.copy(slot.pos);
      slot.vel.y -= 26.0 * dt;
      _droppingNextPos.copy(slot.pos).addScaledVector(slot.vel, dt);
      _droppingSegDir.subVectors(_droppingNextPos, slot.prevPos);
      const segLen = _droppingSegDir.length();

      if (segLen > 1e-5) {
        _droppingSegDir.divideScalar(segLen);

        let rayHit = this.physics.raycastGround(
          slot.prevPos,
          _droppingSegDir,
          segLen + 0.04,
          undefined,
          _droppingRayHit
        );
        if (rayHit.hit && this.npcs?.isHumanCollider?.(rayHit.body)) {
          rayHit = this.physics.raycastGround(
            slot.prevPos,
            _droppingSegDir,
            segLen + 0.04,
            rayHit.body ?? undefined,
            _droppingRayHit
          );
        }

        const midX = (slot.prevPos.x + _droppingNextPos.x) * 0.5;
        const midY = (slot.prevPos.y + _droppingNextPos.y) * 0.5;
        const midZ = (slot.prevPos.z + _droppingNextPos.z) * 0.5;
        const basin = this.physics.getBasinAtXZ(midX, midZ, midY);
        let waterHit = false;
        let waterDist = Infinity;
        if (
          basin &&
          slot.prevPos.y >= basin.surfaceY - 0.06 &&
          _droppingNextPos.y <= basin.surfaceY
        ) {
          const denom = Math.max(1e-4, slot.prevPos.y - _droppingNextPos.y);
          const frac = Math.max(0, Math.min(1, (slot.prevPos.y - basin.surfaceY) / denom));
          waterHit = true;
          waterDist = frac * segLen;
        }

        const solidLimitDist = Math.min(
          waterHit ? waterDist : Infinity,
          rayHit.hit ? rayHit.distance : Infinity
        );
        if (solidLimitDist < segLen) {
          _droppingClampedEnd.copy(slot.prevPos).addScaledVector(_droppingSegDir, solidLimitDist);
        } else {
          _droppingClampedEnd.copy(_droppingNextPos);
        }

        const npcHit =
          this.npcs?.checkFallingDroppingHit?.(slot.prevPos, _droppingClampedEnd, this.position) ??
          null;

        if (npcHit?.hit) {
          this.fx.placeDroppingSplat(npcHit.point, npcHit.normal, npcHit.attachParent);
          this.audio.playDroppingSplat('human', npcHit.point);
          slot.active = false;
          slot.mesh.visible = false;
          continue;
        }

        if (waterHit && (!rayHit.hit || waterDist <= rayHit.distance)) {
          _droppingWaterPt.copy(slot.prevPos).addScaledVector(_droppingSegDir, waterDist);
          _droppingWaterPt.y = basin!.surfaceY;
          this.fx.spawnWaterRipple(_droppingWaterPt, 1.85);
          this.audio.playDroppingSplat('water', _droppingWaterPt);
          slot.active = false;
          slot.mesh.visible = false;
          continue;
        }

        if (rayHit.hit) {
          const hitBody = rayHit.body;
          const attachParent = this.physics.getBodyAttachTarget(hitBody);
          if (hitBody && hitBody.mass > 0) {
            hitBody.wakeUp();
            const tap = Math.min(1.1, 0.45 / Math.max(0.2, hitBody.mass));
            hitBody.velocity.y -= tap;
          }
          this.fx.placeDroppingSplat(rayHit.point, rayHit.normal, attachParent);
          this.audio.playDroppingSplat('surface', rayHit.point);
          slot.active = false;
          slot.mesh.visible = false;
          continue;
        }
      }

      slot.pos.copy(_droppingNextPos);
      slot.mesh.position.copy(slot.pos);
      _droppingTrailDir.copy(slot.vel);
      if (_droppingTrailDir.lengthSq() > 1e-4) {
        _droppingTrailDir.normalize().negate();
        slot.mesh.quaternion.setFromUnitVectors(_droppingUpAxis, _droppingTrailDir);
      }
    }
  }

  private updateHeadLookIK(dt: number): void {
    const smoothRate = Math.min(1, dt * 16.0);
    if (this.introLookTarget) {
      _localTarget.copy(this.introLookTarget).sub(this.position);
      const sinY = Math.sin(-this.yaw);
      const cosY = Math.cos(-this.yaw);
      const lx = _localTarget.x * cosY + _localTarget.z * sinY;
      const lz = -_localTarget.x * sinY + _localTarget.z * cosY;
      if (lz > -0.15) {
        const yawAngle = Math.max(-0.55, Math.min(0.55, Math.atan2(lx, Math.max(0.25, lz)) * 0.72));
        const pitchAngle = Math.max(
          -0.28,
          Math.min(0.28, -Math.atan2(_localTarget.y - 0.18, Math.hypot(lx, lz)) * 0.65)
        );
        _targetHeadLook.set(yawAngle, pitchAngle);
        this.headLookLocal.lerp(_targetHeadLook, smoothRate);
        return;
      }
    }

    let bestTarget: THREE.Vector3 | null = null;
    let bestDist = 5.5;

    if (!this.carriedProp) {
      const props = this.physics.props;
      for (let i = 0; i < props.length; i++) {
        const p = props[i];
        if (p.consumed || p.waterRespawnTimer !== undefined || !p.grabbable) continue;
        const d = p.mesh.position.distanceTo(this.position);
        if (d < bestDist) {
          bestDist = d;
          bestTarget = p.mesh.position;
        }
      }
    }

    const extras = this.getExtraAttentionTargets?.();
    if (extras) {
      for (let i = 0; i < extras.length; i++) {
        const ex = extras[i];
        const d = ex.position.distanceTo(this.position) / Math.max(0.5, ex.priority);
        if (d < bestDist) {
          bestDist = d;
          bestTarget = ex.position;
        }
      }
    }

    if (bestTarget && this.mode !== 'flying') {
      _localTarget.copy(bestTarget).sub(this.position);
      const sinY = Math.sin(-this.yaw);
      const cosY = Math.cos(-this.yaw);
      const lx = _localTarget.x * cosY + _localTarget.z * sinY;
      const lz = -_localTarget.x * sinY + _localTarget.z * cosY;

      if (lz > -0.2) {
        const yawAngle = Math.max(-0.68, Math.min(0.68, Math.atan2(lx, Math.max(0.2, lz))));
        const pitchAngle = Math.max(
          -0.45,
          Math.min(0.45, -Math.atan2(_localTarget.y - 0.2, Math.hypot(lx, lz)))
        );
        _targetHeadLook.set(yawAngle, pitchAngle);
        this.headLookLocal.lerp(_targetHeadLook, smoothRate);
        return;
      }
    }

    _targetHeadLook.set(0, 0);
    this.headLookLocal.lerp(_targetHeadLook, smoothRate);
  }
}
