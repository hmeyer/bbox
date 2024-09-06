# bbox
![test workflow](https://github.com/hmeyer/bbox/actions/workflows/test.yml/badge.svg?branch=master)
![build workflow](https://github.com/hmeyer/bbox/actions/workflows/build.yml/badge.svg?branch=master)
[![Cargo](https://img.shields.io/crates/v/bbox.svg)](https://crates.io/crates/bbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/crates/d/bbox.svg)](#downloads)


bbox is crate for managing axis aligned Bounding Boxes.
Bounding Boxes can be created, dilated, transformed and joined with other Bounding Boxes using
CSG operations.
Finally you can test whether or not a Bounding Box contains some point and what approximate
distance a Point has to the Box.
# Examples

Insert points into a Bounding Box:

```rust
use nalgebra as na;
let mut bbox = bbox::BoundingBox::neg_infinity();
bbox.insert(&na::Point::from([0., 0., 0.]));
bbox.insert(&na::Point::from([1., 2., 3.]));

// or insert multiple points at once

let bbox = bbox::BoundingBox::from([na::Point::from([0., 0., 0.]),
                                    na::Point::from([1., 2., 3.])]);
```
Intersect two Bounding Boxes:

```rust
use nalgebra as na;
let bbox1 = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
                                   &na::Point::from([1., 2., 3.]));
let bbox2 = bbox::BoundingBox::new(&na::Point::from([-1., -2., -3.]),
                                   &na::Point::from([3., 2., 1.]));
let intersection = bbox1.intersection(&bbox2);
```
Rotate a Bounding Box:

```rust
use nalgebra as na;
let rotation = na::Rotation::from_euler_angles(10., 11., 12.);
let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
                                  &na::Point::from([1., 2., 3.]));
let rotated_box = bbox.transform(&rotation.to_homogeneous());
```
Is a point contained in the Box?

```rust
use nalgebra as na;
let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
                                  &na::Point::from([1., 2., 3.]));
let result = bbox.contains(&na::Point::from([1., 1., 1.]));
```
Calculate approximate distance of a point to the Box:

```rust
use nalgebra as na;
let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
                                  &na::Point::from([1., 2., 3.]));
let distance = bbox.distance(&na::Point::from([1., 1., 1.]));
```
## Cargo Features

* `mint` - Enable interoperation with other math libraries through the
  [`mint`](https://crates.io/crates/mint) interface.

#### License

<sup>
Licensed under the <a href="LICENSE">MIT license</a>.
</sup>
