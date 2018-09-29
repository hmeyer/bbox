//! bbox is crate for managing axis aligned 3d Bounding Boxes.
//! Bounding Boxes can be created, dilated, transformed and joined with other Bounding Boxes using
//! CSG operations.
//! Finally you can test whether or not a Bounding Box contains some point and what approximate
//! distance a Point has to the Box.
//! # Examples
//!
//! Intersect two Bounding Boxes:
//!
//! ```rust,no_run
//! extern crate nalgebra as na;
//! extern crate bbox;
//! let bbox1 = bbox::BoundingBox::<f64>::new(&na::Point3::new(0., 0., 0.),
//!                                           &na::Point3::new(1., 2., 3.));
//! let bbox2 = bbox::BoundingBox::<f64>::new(&na::Point3::new(-1., -2., -3.),
//!                                           &na::Point3::new(3., 2., 1.));
//! let intersection = bbox1.intersection(&bbox2);
//! ```
//! Rotate a Bounding Box:
//!
//! ```rust
//! extern crate nalgebra as na;
//! extern crate bbox;
//! let rotation = na::Rotation::from_euler_angles(10., 11., 12.).to_homogeneous();
//! let bbox = bbox::BoundingBox::<f64>::new(&na::Point3::new(0., 0., 0.),
//!                                          &na::Point3::new(1., 2., 3.));
//! let rotated_box = bbox.transform(&rotation);
//! ```
//! Is a point contained in the Box?
//!
//! ```rust
//! extern crate nalgebra as na;
//! extern crate bbox;
//! let bbox = bbox::BoundingBox::<f64>::new(&na::Point3::new(0., 0., 0.),
//!                                          &na::Point3::new(1., 2., 3.));
//! let result = bbox.contains(&na::Point3::new(1., 1., 1.));
//! ```
//! Calculate approximate distance of a point to the Box:
//!
//! ```rust
//! extern crate nalgebra as na;
//! extern crate bbox;
//! let bbox = bbox::BoundingBox::<f64>::new(&na::Point3::new(0., 0., 0.),
//!                                          &na::Point3::new(1., 2., 3.));
//! let distance = bbox.distance(&na::Point3::new(1., 1., 1.));
//! ```
#![warn(missing_docs)]
extern crate alga;
extern crate nalgebra as na;
extern crate num_traits;

use alga::general::Real;
use num_traits::Float;
use std::fmt::Debug;

/// 3D Bounding Box - defined by two diagonally opposing points.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox<S: Real + Debug> {
    /// X-Y-Z-Minimum corner of the box.
    pub min: na::Point3<S>,
    /// X-Y-Z-Maximum corner of the box.
    pub max: na::Point3<S>,
}

fn point_min<S: Real + Float>(p: &[na::Point3<S>]) -> na::Point3<S> {
    p.iter().fold(
        na::Point3::<S>::new(S::infinity(), S::infinity(), S::infinity()),
        |mut min, current| {
            min.x = Real::min(min.x, current.x);
            min.y = Real::min(min.y, current.y);
            min.z = Real::min(min.z, current.z);
            min
        },
    )
}
fn point_max<S: Real + Float>(p: &[na::Point3<S>]) -> na::Point3<S> {
    p.iter().fold(
        na::Point3::<S>::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
        |mut max, current| {
            max.x = Real::max(max.x, current.x);
            max.y = Real::max(max.y, current.y);
            max.z = Real::max(max.z, current.z);
            max
        },
    )
}

impl<S: Float + Real> BoundingBox<S> {
    /// Returns an infinte sized box.
    pub fn infinity() -> BoundingBox<S> {
        BoundingBox {
            min: na::Point3::<S>::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
            max: na::Point3::<S>::new(S::infinity(), S::infinity(), S::infinity()),
        }
    }
    /// Returns a negatively infinte sized box.
    pub fn neg_infinity() -> BoundingBox<S> {
        BoundingBox {
            min: na::Point3::<S>::new(S::infinity(), S::infinity(), S::infinity()),
            max: na::Point3::<S>::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
        }
    }
    /// Create a new Bounding Box by supplying two points.
    pub fn new(a: &na::Point3<S>, b: &na::Point3<S>) -> BoundingBox<S> {
        BoundingBox {
            min: na::Point3::<S>::new(
                Real::min(a.x, b.x),
                Real::min(a.y, b.y),
                Real::min(a.z, b.z),
            ),
            max: na::Point3::<S>::new(
                Real::max(a.x, b.x),
                Real::max(a.y, b.y),
                Real::max(a.z, b.z),
            ),
        }
    }
    /// Create a CSG Union of two Bounding Boxes.
    pub fn union(&self, other: &BoundingBox<S>) -> BoundingBox<S> {
        BoundingBox {
            min: point_min(&[self.min, other.min]),
            max: point_max(&[self.max, other.max]),
        }
    }
    /// Create a CSG Intersection of two Bounding Boxes.
    pub fn intersection(&self, other: &BoundingBox<S>) -> BoundingBox<S> {
        BoundingBox {
            min: point_max(&[self.min, other.min]),
            max: point_min(&[self.max, other.max]),
        }
    }
    /// Transform a Bounding Box - resulting in a enclosing axis aligned Bounding Box.
    pub fn transform<M>(&self, mat: &M) -> BoundingBox<S>
    where
        M: alga::linear::Transformation<na::Point3<S>>,
    {
        let a = &self.min;
        let b = &self.max;
        let corners = [
            mat.transform_point(&na::Point3::<S>::new(a.x, a.y, a.z)),
            mat.transform_point(&na::Point3::<S>::new(a.x, a.y, b.z)),
            mat.transform_point(&na::Point3::<S>::new(a.x, b.y, a.z)),
            mat.transform_point(&na::Point3::<S>::new(a.x, b.y, b.z)),
            mat.transform_point(&na::Point3::<S>::new(b.x, a.y, a.z)),
            mat.transform_point(&na::Point3::<S>::new(b.x, a.y, b.z)),
            mat.transform_point(&na::Point3::<S>::new(b.x, b.y, a.z)),
            mat.transform_point(&na::Point3::<S>::new(b.x, b.y, b.z)),
        ];
        BoundingBox {
            min: point_min(&corners),
            max: point_max(&corners),
        }
    }
    /// Dilate a Bounding Box by some amount in all directions.
    pub fn dilate(&mut self, d: S) -> &mut Self {
        self.min.x -= d;
        self.min.y -= d;
        self.min.z -= d;
        self.max.x += d;
        self.max.y += d;
        self.max.z += d;
        self
    }
    /// Add a Point to a Bounding Box, e.g. expand the Bounding Box to contain that point.
    pub fn insert(&mut self, o: &na::Point3<S>) -> &mut Self {
        self.min.x = Real::min(self.min.x, o.x);
        self.min.y = Real::min(self.min.y, o.y);
        self.min.z = Real::min(self.min.z, o.z);
        self.max.x = Real::max(self.max.x, o.x);
        self.max.y = Real::max(self.max.y, o.y);
        self.max.z = Real::max(self.max.z, o.z);
        self
    }
    /// Return the size of the Box.
    pub fn dim(&self) -> na::Vector3<S> {
        self.max - self.min
    }
    /// Returns the approximate distance of p to the box. The result is guarateed to be not less
    /// than the euclidean distance of p to the box.
    pub fn distance(&self, p: &na::Point3<S>) -> S {
        // If p is not inside (neg), then it is outside (pos) on only one side.
        // So so calculating the max of the diffs on both sides should result in the true value,
        // if positive.
        let xval = Real::max(p.x - self.max.x, self.min.x - p.x);
        let yval = Real::max(p.y - self.max.y, self.min.y - p.y);
        let zval = Real::max(p.z - self.max.z, self.min.z - p.z);
        Real::max(xval, Real::max(yval, zval))
    }
    /// Return true if the Bounding Box contains p.
    pub fn contains(&self, p: &na::Point3<S>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }
}
