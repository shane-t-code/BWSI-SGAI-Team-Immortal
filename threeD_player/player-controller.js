import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

/**
 * PlayerController
 * A free-moving first-person controller for Three.js.
 * Attach it to any scene + camera; call update(delta) every frame.
 */
export class PlayerController {
  constructor(camera, domElement, scene, options = {}) {
    this.camera = camera;
    this.scene = scene;

    // --- Tunable settings ---
    this.eyeHeight = options.eyeHeight ?? 1.7;
    this.moveSpeed = options.moveSpeed ?? 6;      // units/sec
    this.sprintMultiplier = options.sprintMultiplier ?? 1.8;
    this.jumpStrength = options.jumpStrength ?? 7;
    this.gravity = options.gravity ?? 18;
    this.groundY = options.groundY ?? 0; // flat map ground height

    // --- Pointer lock look controls ---
    this.controls = new PointerLockControls(camera, domElement);
    this.scene.add(this.controls.getObject());
    this.camera.position.set(0, this.eyeHeight, 0);

    // Click to lock the mouse (standard FPS behavior)
    domElement.addEventListener('click', () => this.controls.lock());

    // --- Input state ---
    this.input = { forward: false, back: false, left: false, right: false, sprint: false };
    this.velocity = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.canJump = true;

    this._bindInput();
  }

  _bindInput() {
    const onKeyDown = (e) => {
      switch (e.code) {
        case 'KeyW': case 'ArrowUp': this.input.forward = true; break;
        case 'KeyS': case 'ArrowDown': this.input.back = true; break;
        case 'KeyA': case 'ArrowLeft': this.input.left = true; break;
        case 'KeyD': case 'ArrowRight': this.input.right = true; break;
        case 'ShiftLeft': case 'ShiftRight': this.input.sprint = true; break;
        case 'Space':
          if (this.canJump) {
            this.velocity.y = this.jumpStrength;
            this.canJump = false;
          }
          break;
      }
    };

    const onKeyUp = (e) => {
      switch (e.code) {
        case 'KeyW': case 'ArrowUp': this.input.forward = false; break;
        case 'KeyS': case 'ArrowDown': this.input.back = false; break;
        case 'KeyA': case 'ArrowLeft': this.input.left = false; break;
        case 'KeyD': case 'ArrowRight': this.input.right = false; break;
        case 'ShiftLeft': case 'ShiftRight': this.input.sprint = false; break;
      }
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
  }

  isLocked() {
    return this.controls.isLocked;
  }

  update(delta) {
    if (!this.controls.isLocked) return;

    // --- Horizontal damping (friction) ---
    this.velocity.x -= this.velocity.x * 10.0 * delta;
    this.velocity.z -= this.velocity.z * 10.0 * delta;

    // --- Gravity ---
    this.velocity.y -= this.gravity * delta;

    // --- Movement intent ---
    this.direction.z = Number(this.input.forward) - Number(this.input.back);
    this.direction.x = Number(this.input.right) - Number(this.input.left);
    this.direction.normalize();

    const speed = this.moveSpeed * (this.input.sprint ? this.sprintMultiplier : 1);

    if (this.input.forward || this.input.back) this.velocity.z -= this.direction.z * speed * delta * 10;
    if (this.input.left || this.input.right) this.velocity.x -= this.direction.x * speed * delta * 10;

    // Apply horizontal movement relative to look direction
    this.controls.moveRight(-this.velocity.x * delta);
    this.controls.moveForward(-this.velocity.z * delta);

    // Apply vertical movement (jump/gravity)
    const obj = this.controls.getObject();
    obj.position.y += this.velocity.y * delta;

    // --- Simple ground collision (flat map) ---
    const floorLevel = this.groundY + this.eyeHeight;
    if (obj.position.y < floorLevel) {
      this.velocity.y = 0;
      obj.position.y = floorLevel;
      this.canJump = true;
    }
  }
}