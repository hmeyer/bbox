//! bbox is crate for managing axis aligned 3d Bounding Boxes.
//! Bounding Boxes can be created, dilated, transformed and joined with other Bounding Boxes using
//! CSG operations.
//! Finally you can test whether or not a Bounding Box contains some point and what approximate
//! distance a Point has to the Box.
//! # Examples
//!
//! Insert points into a Bounding Box:
//!
//! ```rust
//! use nalgebra as na;
//! let mut bbox = bbox::BoundingBox::neg_infinity();
//! bbox.insert(&na::Point::from([0., 0., 0.]));
//! bbox.insert(&na::Point::from([1., 2., 3.]));
//!
//! // or insert multiple points at once
//!
//! let bbox = bbox::BoundingBox::from([na::Point::from([0., 0., 0.]),
//!                                            na::Point::from([1., 2., 3.])]);
//! ```
//! Intersect two Bounding Boxes:
//!
//! ```rust
//! use nalgebra as na;
//! let bbox1 = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                           &na::Point::from([1., 2., 3.]));
//! let bbox2 = bbox::BoundingBox::new(&na::Point::from([-1., -2., -3.]),
//!                                           &na::Point::from([3., 2., 1.]));
//! let intersection = bbox1.intersection(&bbox2);
//! ```
//! Rotate a Bounding Box:
//!
//! ```rust
//! use nalgebra as na;
//! let rotation = na::Rotation::from_euler_angles(10., 11., 12.);
//! let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                          &na::Point::from([1., 2., 3.]));
//! let rotated_box = bbox.transform(&rotation.to_homogeneous());
//! ```
//! Is a point contained in the Box?
//!
//! ```rust
//! use nalgebra as na;
//! let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                          &na::Point::from([1., 2., 3.]));
//! let result = bbox.contains(&na::Point::from([1., 1., 1.]));
//! ```
//! Calculate approximate distance of a point to the Box:
//!
//! ```rust
//! use nalgebra as na;
//! let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                          &na::Point::from([1., 2., 3.]));
//! let distance = bbox.distance(&na::Point::from([1., 1., 1.]));
//! ```
//! ## Cargo Features
//!
//! * `mint` - Enable interoperation with other math libraries through the
//!   [`mint`](https://crates.io/crates/mint) interface.

#![warn(missing_docs)]

use approx::{AbsDiffEq, RelativeEq};
use nalgebra as na;
use num_traits::Float;
use std::fmt::Debug;

fn points_best<S: 'static + Float + Debug, const D: usize>(
    p: &[na::Point<S, D>],
    op: fn(S, S) -> S,
) -> na::Point<S, D> {
    p.iter().fold(
        na::Point::from([-op(S::infinity(), S::neg_infinity()); D]),
        |mut best, current| {
            for (best_i, current_i) in best.iter_mut().zip(current.iter()) {
                *best_i = op(*best_i, *current_i);
            }
            best
        },
    )
}

fn points_min<S: 'static + Float + Debug, const D: usize>(
    p: &[na::Point<S, D>],
) -> na::Point<S, D> {
    points_best(p, S::min)
}

fn points_max<S: 'static + Float + Debug, const D: usize>(
    p: &[na::Point<S, D>],
) -> na::Point<S, D> {
    points_best(p, S::max)
}

/// 3D Bounding Box - defined by two diagonally opposing points.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox<S: 'static + Debug + Copy + PartialEq> {
    /// X-Y-Z-Minimum corner of the box.
    pub min: na::Point<S, 3>,
    /// X-Y-Z-Maximum corner of the box.
    pub max: na::Point<S, 3>,
}

impl<S: Float + na::RealField, T: AsRef<[na::Point<S, 3>]>> From<T> for BoundingBox<S> {
    fn from(points: T) -> Self {
        Self {
            min: points_min(points.as_ref()),
            max: points_max(points.as_ref()),
        }
    }
}

impl<S: Float + Debug + na::RealField + simba::scalar::RealField> BoundingBox<S> {
    /// Returns an infinte sized box.
    pub fn infinity() -> Self {
        Self {
            min: na::Point::from([S::neg_infinity(), S::neg_infinity(), S::neg_infinity()]),
            max: na::Point::from([S::infinity(), S::infinity(), S::infinity()]),
        }
    }
    /// Returns a negatively infinte sized box.
    pub fn neg_infinity() -> Self {
        Self {
            min: na::Point::from([S::infinity(), S::infinity(), S::infinity()]),
            max: na::Point::from([S::neg_infinity(), S::neg_infinity(), S::neg_infinity()]),
        }
    }
    /// Create a new Bounding Box by supplying two points.
    pub fn new(a: &na::Point<S, 3>, b: &na::Point<S, 3>) -> Self {
        Self {
            min: na::Point::from([
                Float::min(a.x, b.x),
                Float::min(a.y, b.y),
                Float::min(a.z, b.z),
            ]),
            max: na::Point::from([
                Float::max(a.x, b.x),
                Float::max(a.y, b.y),
                Float::max(a.z, b.z),
            ]),
        }
    }
    /// Returns true if the Bounding Box is empty.
    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }
    /// Returns true if the Bounding Box has finite size.
    pub fn is_finite(&self) -> bool {
        self.min.x.is_finite()
            && self.min.y.is_finite()
            && self.min.z.is_finite()
            && self.max.x.is_finite()
            && self.max.y.is_finite()
            && self.max.z.is_finite()
    }
    /// Returns the center point of the Bounding Box.
    pub fn center(&self) -> na::Point<S, 3> {
        na::Point::from([
            (self.min.x + self.max.x) / S::from(2.0).unwrap(),
            (self.min.y + self.max.y) / S::from(2.0).unwrap(),
            (self.min.z + self.max.z) / S::from(2.0).unwrap(),
        ])
    }
    /// Create a CSG Union of two Bounding Boxes.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: points_min(&[self.min, other.min]),
            max: points_max(&[self.max, other.max]),
        }
    }
    /// Create a CSG Intersection of two Bounding Boxes.
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: points_max(&[self.min, other.min]),
            max: points_min(&[self.max, other.max]),
        }
    }
    /// Get the corners of the Bounding Box
    pub fn get_corners(&self) -> [na::Point<S, 3>; 8] {
        [
            na::Point::from([self.min.x, self.min.y, self.min.z]),
            na::Point::from([self.min.x, self.min.y, self.max.z]),
            na::Point::from([self.min.x, self.max.y, self.min.z]),
            na::Point::from([self.min.x, self.max.y, self.max.z]),
            na::Point::from([self.max.x, self.min.y, self.min.z]),
            na::Point::from([self.max.x, self.min.y, self.max.z]),
            na::Point::from([self.max.x, self.max.y, self.min.z]),
            na::Point::from([self.max.x, self.max.y, self.max.z]),
        ]
    }
    /// Transform a Bounding Box - resulting in a enclosing axis aligned Bounding Box.
    pub fn transform(&self, mat: &na::Matrix4<S>) -> Self {
        let corners = self.get_corners();
        let transformed: Vec<_> = corners
            .into_iter()
            .map(|c| mat.transform_point(&c))
            .collect();
        Self::from(&transformed)
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
    pub fn insert(&mut self, o: &na::Point<S, 3>) -> &mut Self {
        self.min.x = Float::min(self.min.x, o.x);
        self.min.y = Float::min(self.min.y, o.y);
        self.min.z = Float::min(self.min.z, o.z);
        self.max.x = Float::max(self.max.x, o.x);
        self.max.y = Float::max(self.max.y, o.y);
        self.max.z = Float::max(self.max.z, o.z);
        self
    }
    /// Return the size of the Box.
    pub fn dim(&self) -> na::Vector3<S> {
        self.max - self.min
    }
    /// Returns the approximate distance of p to the box. The result is guarateed to be not less
    /// than the euclidean distance of p to the box.
    pub fn distance(&self, p: &na::Point<S, 3>) -> S {
        // If p is not inside (neg), then it is outside (pos) on only one side.
        // So so calculating the max of the diffs on both sides should result in the true value,
        // if positive.
        let xval = Float::max(p.x - self.max.x, self.min.x - p.x);
        let yval = Float::max(p.y - self.max.y, self.min.y - p.y);
        let zval = Float::max(p.z - self.max.z, self.min.z - p.z);
        Float::max(xval, Float::max(yval, zval))
    }
    /// Return true if the Bounding Box contains p.
    pub fn contains(&self, p: &na::Point<S, 3>) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

impl<T: Float> AbsDiffEq for BoundingBox<T>
where
    <T as AbsDiffEq>::Epsilon: Copy,
    T: AbsDiffEq + Debug,
{
    type Epsilon = <T as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        <T as AbsDiffEq>::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        na::Point::abs_diff_eq(&self.min, &other.min, epsilon)
            && na::Point::abs_diff_eq(&self.max, &other.max, epsilon)
    }
}

impl<T: Float> RelativeEq for BoundingBox<T>
where
    <T as AbsDiffEq>::Epsilon: Copy,
    T: RelativeEq + Debug,
{
    fn default_max_relative() -> <T as AbsDiffEq>::Epsilon {
        <T as RelativeEq>::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: <T as AbsDiffEq>::Epsilon,
        max_relative: <T as AbsDiffEq>::Epsilon,
    ) -> bool {
        na::Point::relative_eq(&self.min, &other.min, epsilon, max_relative)
            && na::Point::relative_eq(&self.max, &other.max, epsilon, max_relative)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn box_contains_points_inside() {
        let bbox = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([1., 2., 3.]),
        );
        assert!(bbox.contains(&na::Point::from([0., 0., 0.])));
        assert!(bbox.contains(&na::Point::from([0., 1., 0.])));
        assert!(bbox.contains(&na::Point::from([1., 0., 1.])));
        assert!(bbox.contains(&na::Point::from([1., 1., 1.])));
        assert!(!bbox.contains(&na::Point::from([2., 2., 2.])));
        assert!(!bbox.contains(&na::Point::from([-1., -1., -1.])));
    }

    #[test]
    fn box_from_points() {
        let points = [
            na::Point::from([0., 0., 0.]),
            na::Point::from([1., 1., 0.]),
            na::Point::from([0., -2., 2.]),
        ];
        for bbox in [
            BoundingBox::from(points),            // from array
            BoundingBox::from(&points[..]),       // from slice
            BoundingBox::from(Vec::from(points)), // from vector
        ] {
            assert_relative_eq!(bbox.min, na::Point::from([0., -2., 0.]));
            assert_relative_eq!(bbox.max, na::Point::from([1., 1., 2.]));
        }
    }

    #[test]
    fn get_corners() {
        let bbox = BoundingBox::new(
            &na::Point::from([1., 2., 3.]),
            &na::Point::from([4., 5., 6.]),
        );
        let corners = bbox.get_corners();
        assert!(corners.contains(&na::Point::from([1., 2., 3.])));
        assert!(corners.contains(&na::Point::from([1., 2., 6.])));
        assert!(corners.contains(&na::Point::from([1., 5., 3.])));
        assert!(corners.contains(&na::Point::from([1., 5., 6.])));
        assert!(corners.contains(&na::Point::from([4., 2., 3.])));
        assert!(corners.contains(&na::Point::from([4., 2., 6.])));
        assert!(corners.contains(&na::Point::from([4., 5., 3.])));
        assert!(corners.contains(&na::Point::from([4., 5., 6.])));
    }

    #[test]
    fn is_empty() {
        let bbox = BoundingBox::<f64>::neg_infinity();
        assert!(bbox.is_empty());
        let bbox = BoundingBox::from([na::Point::from([0., 0., 0.])]);
        assert!(!bbox.is_empty());
    }

    #[test]
    fn is_finite() {
        let bbox = BoundingBox::<f64>::neg_infinity();
        assert!(!bbox.is_finite());
        let bbox = BoundingBox::from([na::Point::from([0., 0., 0.])]);
        assert!(bbox.is_finite());
        let bbox = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([f64::INFINITY, 1., 1.]),
        );
        assert!(!bbox.is_finite());
    }

    #[test]
    fn center() {
        let bbox = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([1., 1., 1.]),
        );
        assert_relative_eq!(bbox.center(), na::Point::from([0.5, 0.5, 0.5]));
    }

    #[test]
    fn transform_with_translation() {
        let bbox = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([1., 1., 1.]),
        );
        assert_relative_eq!(
            bbox.transform(&na::Translation3::new(1., 2., 3.).to_homogeneous()),
            BoundingBox::new(
                &na::Point::from([1., 2., 3.]),
                &na::Point::from([2., 3., 4.]),
            )
        );
    }

    #[test]
    fn transform_with_rotation() {
        let bbox = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([1., 1., 1.]),
        );
        assert_relative_eq!(
            bbox.transform(
                &na::Rotation::from_euler_angles(::std::f64::consts::PI / 2., 0., 0.)
                    .to_homogeneous()
            ),
            BoundingBox::new(
                &na::Point::from([0., -1., 0.]),
                &na::Point::from([1., 0., 1.]),
            )
        );
    }

    #[test]
    fn union_of_two_boxes() {
        let bbox1 = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([4., 8., 16.]),
        );
        let bbox2 = BoundingBox::new(
            &na::Point::from([2., 2., 2.]),
            &na::Point::from([16., 4., 8.]),
        );
        assert_relative_eq!(
            bbox1.union(&bbox2),
            BoundingBox::new(
                &na::Point::from([0., 0., 0.]),
                &na::Point::from([16., 8., 16.]),
            )
        );
    }

    #[test]
    fn intersection_of_two_boxes() {
        let bbox1 = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([4., 8., 16.]),
        );
        let bbox2 = BoundingBox::new(
            &na::Point::from([2., 2., 2.]),
            &na::Point::from([16., 4., 8.]),
        );
        assert_relative_eq!(
            bbox1.intersection(&bbox2),
            BoundingBox::new(
                &na::Point::from([2., 2., 2.]),
                &na::Point::from([4., 4., 8.]),
            )
        );
    }

    #[test]
    fn dilate() {
        let mut bbox = BoundingBox::new(
            &na::Point::from([0., 0., 0.]),
            &na::Point::from([1., 1., 1.]),
        );
        assert_relative_eq!(
            bbox.dilate(0.1),
            &mut BoundingBox::new(
                &na::Point::from([-0.1, -0.1, -0.1]),
                &na::Point::from([1.1, 1.1, 1.1]),
            )
        );
        assert_relative_eq!(
            bbox.dilate(-0.5),
            &mut BoundingBox::new(
                &na::Point::from([0.4, 0.4, 0.4]),
                &na::Point::from([0.6, 0.6, 0.6]),
            )
        );
    }

    #[test]
    fn box_contains_inserted_points() {
        let mut bbox = BoundingBox::neg_infinity();
        let p1 = na::Point::from([1., 0., 0.]);
        let p2 = na::Point::from([0., 2., 3.]);
        assert!(!bbox.contains(&p1));
        bbox.insert(&p1);
        assert!(bbox.contains(&p1));
        assert!(!bbox.contains(&p2));
        bbox.insert(&p2);
        assert!(bbox.contains(&p2));
    }
}
