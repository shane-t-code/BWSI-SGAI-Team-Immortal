import * as THREE from 'three';

/**
 * Fuse - Represents a single fuse object in the 3D world
 */
class Fuse {
  constructor(id, position) {
    this.id = id;
    this.position = position.clone();
    this.mesh = this.createMesh();
    this.mesh.position.copy(position);
    this.mesh.userData = { type: 'fuse', fuseId: id };
  }

  createMesh() {
    // Create a realistic fuse with detailed geometry
    const group = new THREE.Group();

    // Main fuse body (cylinder with more segments)
    const bodyGeometry = new THREE.CylinderGeometry(0.08, 0.08, 0.4, 16);
    const bodyMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x4a3728,
      roughness: 0.6,
      metalness: 0.3
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.2;
    group.add(body);

    // Fuse casing (metal ring at bottom)
    const casingGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.08, 16);
    const casingMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x888888,
      roughness: 0.3,
      metalness: 0.8
    });
    const casing = new THREE.Mesh(casingGeometry, casingMaterial);
    casing.position.y = 0.04;
    group.add(casing);

    // Fuse tip (glowing spark)
    const tipGeometry = new THREE.SphereGeometry(0.06, 16, 16);
    const tipMaterial = new THREE.MeshStandardMaterial({ 
      color: 0xFF6600,
      emissive: 0xFF4500,
      emissiveIntensity: 2
    });
    const tip = new THREE.Mesh(tipGeometry, tipMaterial);
    tip.position.y = 0.42;
    group.add(tip);

    // Spark particles (small glowing dots around tip)
    for (let i = 0; i < 5; i++) {
      const sparkGeometry = new THREE.SphereGeometry(0.015, 8, 8);
      const sparkMaterial = new THREE.MeshStandardMaterial({ 
        color: 0xFFAA00,
        emissive: 0xFF6600,
        emissiveIntensity: 1.5
      });
      const spark = new THREE.Mesh(sparkGeometry, sparkMaterial);
      const angle = (i / 5) * Math.PI * 2;
      const radius = 0.08;
      spark.position.set(
        Math.cos(angle) * radius,
        0.45 + Math.random() * 0.05,
        Math.sin(angle) * radius
      );
      group.add(spark);
    }

    // Add point light for glow effect
    const light = new THREE.PointLight(0xFF6600, 1.5, 4);
    light.position.y = 0.45;
    group.add(light);

    return group;
  }

  addToScene(scene) {
    scene.add(this.mesh);
  }

  removeFromScene(scene) {
    scene.remove(this.mesh);
  }

  setPosition(position) {
    this.position.copy(position);
    this.mesh.position.copy(position);
  }
}

/**
 * FuseSystem - Manages fuse spawning, pickup, drop, and inventory
 */
export class FuseSystem {
  constructor(scene, camera, playerController) {
    this.scene = scene;
    this.camera = camera;
    this.playerController = playerController;
    
    this.fuses = [];
    this.inventory = [];
    this.greenTiles = [];
    this.maxFuses = 3;
    this.pickupDistance = 2.5;
    
    this.raycaster = new THREE.Raycaster();
    this.interactionKey = 'KeyE';
    
    this._generateGreenTiles();
    this._spawnFuses();
    this._bindInput();
  }

  _generateGreenTiles() {
    // Create a grid of green tiles on the ground
    const tileSize = 4;
    const mapSize = 40; // 40x40 area
    const tileCount = Math.floor(mapSize / tileSize);
    
    for (let x = -tileCount / 2; x < tileCount / 2; x++) {
      for (let z = -tileCount / 2; z < tileCount / 2; z++) {
        const posX = x * tileSize;
        const posZ = z * tileSize;
        
        // Create tile geometry
        const tileGeometry = new THREE.PlaneGeometry(tileSize - 0.1, tileSize - 0.1);
        const tileMaterial = new THREE.MeshStandardMaterial({ 
          color: 0x3a9d43,
          roughness: 0.9
        });
        const tile = new THREE.Mesh(tileGeometry, tileMaterial);
        tile.rotation.x = -Math.PI / 2;
        tile.position.set(posX, 0.01, posZ); // Slightly above ground
        tile.userData = { type: 'greenTile' };
        
        this.scene.add(tile);
        this.greenTiles.push({
          mesh: tile,
          position: new THREE.Vector3(posX, 0.01, posZ)
        });
      }
    }
  }

  _spawnFuses() {
    // Spawn 3 fuses on random green tiles
    const shuffledTiles = [...this.greenTiles].sort(() => Math.random() - 0.5);
    
    for (let i = 0; i < Math.min(this.maxFuses, shuffledTiles.length); i++) {
      const tile = shuffledTiles[i];
      const fusePosition = tile.position.clone();
      fusePosition.y = 0.5; // Place fuse above the tile
      
      const fuse = new Fuse(i, fusePosition);
      fuse.addToScene(this.scene);
      this.fuses.push(fuse);
    }
  }

  _bindInput() {
    document.addEventListener('keydown', (e) => {
      if (e.code === this.interactionKey) {
        this.handleInteraction();
      }
    });
  }

  handleInteraction() {
    if (!this.playerController.isLocked()) return;

    // Check for pickup (looking at a fuse)
    const nearbyFuse = this.getNearbyFuse();
    if (nearbyFuse && this.isLookingAtFuse(nearbyFuse)) {
      this.pickupFuse(nearbyFuse);
      return;
    }

    // Check for drop (if has inventory)
    if (this.inventory.length > 0) {
      this.dropFuse();
    }
  }

  getNearbyFuse() {
    const playerPosition = this.playerController.controls.getObject().position;
    
    for (const fuse of this.fuses) {
      const distance = playerPosition.distanceTo(fuse.mesh.position);
      if (distance <= this.pickupDistance) {
        return fuse;
      }
    }
    return null;
  }

  isLookingAtFuse(fuse) {
    // Set up raycaster from camera center
    this.raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    
    // Check if ray intersects with the fuse mesh
    const intersects = this.raycaster.intersectObject(fuse.mesh, true);
    
    return intersects.length > 0;
  }

  pickupFuse(fuse) {
    // Remove from scene
    fuse.removeFromScene(this.scene);
    
    // Remove from fuses array
    const index = this.fuses.indexOf(fuse);
    if (index > -1) {
      this.fuses.splice(index, 1);
    }
    
    // Add to inventory
    this.inventory.push(fuse);
    
    // Update UI
    this.updateInventoryUI();
    
    console.log(`Picked up fuse ${fuse.id}`);
  }

  dropFuse() {
    const fuse = this.inventory.pop();
    if (!fuse) return;
    
    // Get player position and drop in front
    const playerPosition = this.playerController.controls.getObject().position.clone();
    const playerDirection = new THREE.Vector3();
    this.camera.getWorldDirection(playerDirection);
    
    const dropPosition = playerPosition.add(playerDirection.multiplyScalar(2));
    dropPosition.y = 0.5;
    
    // Check if dropped on green tile
    const onGreenTile = this.isOnGreenTile(dropPosition);
    
    if (onGreenTile) {
      fuse.setPosition(dropPosition);
      fuse.addToScene(this.scene);
      this.fuses.push(fuse);
      
      console.log(`Dropped fuse ${fuse.id}`);
    } else {
      // Return to inventory if not on green tile
      this.inventory.push(fuse);
      console.log('Cannot drop fuse - must be on green tile');
    }
    
    this.updateInventoryUI();
  }

  isOnGreenTile(position) {
    for (const tile of this.greenTiles) {
      const distance = new THREE.Vector2(position.x, position.z)
        .distanceTo(new THREE.Vector2(tile.position.x, tile.position.z));
      
      if (distance < 2) { // Within tile radius
        return true;
      }
    }
    return false;
  }

  updateInventoryUI() {
    const inventorySlots = document.getElementById('inventory-slots');
    if (!inventorySlots) return;
    
    inventorySlots.innerHTML = '';
    
    for (let i = 0; i < this.inventory.length; i++) {
      const fuseSlot = document.createElement('div');
      fuseSlot.className = 'fuse-slot';
      fuseSlot.innerHTML = `<span>Fuse ${this.inventory[i].id + 1}</span>`;
      inventorySlots.appendChild(fuseSlot);
    }
    
    // Update count
    const countElement = document.getElementById('fuse-count');
    if (countElement) {
      countElement.textContent = this.inventory.length;
    }
  }

  update(delta) {
    // Animate fuses (floating effect)
    const time = Date.now() * 0.001;
    
    for (const fuse of this.fuses) {
      fuse.mesh.position.y = 0.5 + Math.sin(time * 2 + fuse.id) * 0.1;
      fuse.mesh.rotation.y += delta * 0.5;
    }
  }

  getInventoryCount() {
    return this.inventory.length;
  }

  getTotalFuses() {
    return this.maxFuses;
  }
}
