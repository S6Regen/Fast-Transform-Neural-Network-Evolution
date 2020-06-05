// Fixed filter bank neural network. 
// Sean O'Connor
class FFBNet {
  // vecLen must be 2,4,8,16,32.....
  constructor(vecLen, depth, hash) {
    this.vecLen = vecLen;
    this.depth = depth;
    this.hash = hash;
    this.params = new Float32Array(2 * vecLen * depth);
    for (let i = 0; i < this.params.length; i++) {
      this.params[i] = 0.5;
    }
  }

  recall(result, input) {
    adjustVec(result, input, 1); // const. vector length.
    signFlipVec(result, this.hash); // frequency scramble
    let paIdx = 0; // parameter index
    for (let i = 0; i < this.depth; i++) {
      whtScVec(result, 2.0); // WHT scale *2 for switching losses.
      for (let j = 0; j < this.vecLen; j++) {
        const signBit = result[j] < 0 ? 0 : 1;
        result[j] *= this.params[paIdx + signBit];
        paIdx += 2;
      }
    }
    whtScVec(result, 1.0);
  }
}

// Fast Walsh Hadamard Transform provide your own scaling
function whtScVec(vec, sc) {
  let n = vec.length;
  let hs = 1;
  while (hs < n) {
    let i = 0;
    while (i < n) {
      const j = i + hs;
      while (i < j) {
        var a = vec[i];
        var b = vec[i + hs];
        vec[i] = a + b;
        vec[i + hs] = a - b;
        i += 1;
      }
      i += hs;
    }
    hs += hs;
  }
  scaleVec(vec, vec, sc / sqrt(n));
}

// pseudorandom sign flip of vector elements based on hash 
function signFlipVec(vec, hash) {
  for (let i = 0, n = vec.length; i < n; i++) {
    hash += 0x3C6EF35F;
    hash *= 0x19660D;
    hash &= 0xffffffff;
    if (((hash * 0x9E3779B9) & 0x80000000) === 0) {
      vec[i] = -vec[i];
    }
  }
}

/* // Fast random projection
function rpVec(vec, hash) {
  signFlipVec(vec, hash);
  whtVec(vec);
} */

function scaleVec(rVec, xVec, sc) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] = xVec[i] * sc;
  }
}

function sumSqVec(vec) {
  let sum = 0.0;
  for (let i = 0, n = vec.length; i < n; i++) {
    sum += vec[i] * vec[i];
  }
  return sum;
}

// Adjust variance/sphere radius
function adjustVec(rVec, xVec, scale) {
  let MIN_SQ = 1e-20;
  let adj = scale / sqrt((sumSqVec(xVec) / xVec.length) + MIN_SQ);
  scaleVec(rVec, xVec, adj);
}

// Sum of squared difference cost
function costL2(vec, tar) {
  var cost = 0;
  for (var i = 0; i < vec.length; i++) {
    var e = vec[i] - tar[i];
    cost += e * e;
  }
  return cost;
}

class Mutator {
  constructor(size, precision) {
    this.previous = new Float32Array(size);
    this.pIdx = new Int32Array(size);
    this.precision = precision;
  }
  mutate(vec) {
    for (let i = 0; i < this.previous.length; i++) {
      let rpos = int(random(vec.length));
      let v = vec[rpos];
      this.pIdx[i] = rpos;
      this.previous[i] = v;
      let m = 2 * exp(random(-this.precision, 0));
      if (random() < 0.5) m = -m;
      let vm = v + m;
      if (vm >= 1) vm = v;
      if (vm <= -1) vm = v;
      vec[rpos] = vm;
    }
  }
  undo(vec) {
    for (let i = this.previous.length - 1; i >= 0; i--) {
      vec[this.pIdx[i]] = this.previous[i];
    }
  }
}

// Test with Lissajous curves
let c1;
let ex = [];
let work = new Float32Array(256);
let parentCost = Number.POSITIVE_INFINITY;
let parentNet = new FFBNet(256, 5, 123456);
let mut = new Mutator(10, 25);

function setup() {
  createCanvas(400, 400);
  c1 = color('gold');
  for (let i = 0; i < 8; i++) {
    ex[i] = new Float32Array(256);
  }
  for (let i = 0; i < 127; i++) { // Training data
    let t = i * 2 * PI / 127;
    ex[0][2 * i] = sin(t);
    ex[0][2 * i + 1] = sin(2 * t);
    ex[1][2 * i] = sin(2 * t);
    ex[1][2 * i + 1] = sin(t);
    ex[2][2 * i] = sin(2 * t);
    ex[2][2 * i + 1] = sin(3 * t);
    ex[3][2 * i] = sin(3 * t);
    ex[3][2 * i + 1] = sin(2 * t);
    ex[4][2 * i] = sin(3 * t);
    ex[4][2 * i + 1] = sin(4 * t);
    ex[5][2 * i] = sin(4 * t);
    ex[5][2 * i + 1] = sin(3 * t);
    ex[6][2 * i] = sin(2 * t);
    ex[6][2 * i + 1] = sin(5 * t);
    ex[7][2 * i] = sin(5 * t);
    ex[7][2 * i + 1] = sin(2 * t);
  }
  textSize(16);
}

function draw() {
  background(0);
  loadPixels();
  for (let i = 0; i < 100; i++) {
    mut.mutate(parentNet.params);
    let cost = 0;
    for (let j = 0; j < 8; j++) {
      parentNet.recall(work, ex[j]);
      cost += costL2(work, ex[j]); // autoassociate
    }
    if (cost < parentCost) {
      parentCost = cost;
    } else {
      mut.undo(parentNet.params);
    }
  }
  fill(c1);
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 255; j += 2) {
      set(25 + i * 40 + 18 * ex[i][j], 44 + 18 * ex[i][j + 1]);
    }
  }
  for (let i = 0; i < 8; i++) {
    parentNet.recall(work, ex[i]);
    for (let j = 0; j < 255; j += 2) {
      set(25 + i * 40 + 18 * work[j], 104 + 18 * work[j + 1]);
    }
  }
  updatePixels();
  text("Training Data", 5, 20);
  text("Autoassociative recall", 5, 80);
  text('Iterations: ' + frameCount, 5, 150);
  text('Cost: ' + parentCost.toFixed(3), 5, 170);
}