# stl_io
[![Build Status](https://travis-ci.org/hmeyer/bbox.svg?branch=master)](https://travis-ci.org/hmeyer/bbox) [![Cargo](https://img.shields.io/crates/v/bbox.svg)](https://crates.io/crates/bbox) [![License: GPL-3.0](https://img.shields.io/crates/l/direct-gui.svg)](#license) [![Downloads](https://img.shields.io/crates/d/bbox.svg)](#downloads)


bbox is crate for managing axis aligned 3d Bounding Boxes.
Bounding Boxes can be created, dilated, transformed and joined with other Bounding Boxes using
CSG operations.
Finally you can test whether or not a Bounding Box contains some point and what approximate
distance a Point has to the Box.

# Examples

Intersect two Bounding Boxes
```rust
extern crate nalgebra as na;
extern crate bbox;
let bbox1 = bbox::BoundingBox::<f64>::new(na::Point3::new(0., 0., 0.),
                                          na::Point3::new(1., 2., 3.));
let bbox2 = bbox::BoundingBox::<f64>::new(na::Point3::new(-1., -2., -3.),
                                          na::Point3::new(3., 2., 1.));
let intersection = bbox1.intersection(&bbox2);
```

Rotate a Bounding Box:
```rust
extern crate nalgebra as na;
extern crate bbox;
let rotation = na::Rotation::from_euler_angles(10., 11., 12.).to_homogeneous();
let bbox = bbox::BoundingBox::<f64>::new(na::Point3::new(0., 0., 0.),
                                         na::Point3::new(1., 2., 3.));
let rotated_box = bbox.transform(&rotation);
```
Is a point contained in the Box?
//!
```rust
extern crate nalgebra as na;
extern crate bbox;
let bbox = bbox::BoundingBox::<f64>::new(na::Point3::new(0., 0., 0.),
                                         na::Point3::new(1., 2., 3.));
let result = bbox.contains(na::Point3::new(1., 1., 1.));
```
Calculate approximate distance of a point to the Box:
//!
```rust
extern crate nalgebra as na;
extern crate bbox;
let bbox = bbox::BoundingBox::<f64>::new(na::Point3::new(0., 0., 0.),
                                         na::Point3::new(1., 2., 3.));
let distance = bbox.distance(na::Point3::new(1., 1., 1.));
```
