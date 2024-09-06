//! bbox is crate for managing axis aligned Bounding Boxes.
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
//!                                     na::Point::from([1., 2., 3.])]);
//! ```
//! Intersect two Bounding Boxes:
//!
//! ```rust
//! use nalgebra as na;
//! let bbox1 = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                    &na::Point::from([1., 2., 3.]));
//! let bbox2 = bbox::BoundingBox::new(&na::Point::from([-1., -2., -3.]),
//!                                    &na::Point::from([3., 2., 1.]));
//! let intersection = bbox1.intersection(&bbox2);
//! ```
//! Rotate a Bounding Box:
//!
//! ```rust
//! use nalgebra as na;
//! let rotation = na::Rotation::from_euler_angles(10., 11., 12.);
//! let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                   &na::Point::from([1., 2., 3.]));
//! let rotated_box = bbox.transform(&rotation.to_homogeneous());
//! ```
//! Is a point contained in the Box?
//!
//! ```rust
//! use nalgebra as na;
//! let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                   &na::Point::from([1., 2., 3.]));
//! let result = bbox.contains(&na::Point::from([1., 1., 1.]));
//! ```
//! Calculate approximate distance of a point to the Box:
//!
//! ```rust
//! use nalgebra as na;
//! let bbox = bbox::BoundingBox::new(&na::Point::from([0., 0., 0.]),
//!                                   &na::Point::from([1., 2., 3.]));
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

fn point_best<S: 'static + Float + Debug, const D: usize>(
    a: &na::Point<S, D>,
    b: &na::Point<S, D>,
    op: fn(S, S) -> S,
) -> na::Point<S, D> {
    let mut best: na::Point<S, D> = na::Point::default();
    for (best_i, (a_i, b_i)) in best.iter_mut().zip(a.iter().zip(b.iter())) {
        *best_i = op(*a_i, *b_i);
    }
    best
}

fn points_best<S: 'static + Float + Debug, const D: usize>(
    points: &[na::Point<S, D>],
    op: fn(S, S) -> S,
) -> na::Point<S, D> {
    points.iter().fold(
        na::Point::from([-op(S::infinity(), S::neg_infinity()); D]),
        |best, current| point_best(&best, current, op),
    )
}

fn point_min<S: 'static + Float + Debug, const D: usize>(
    a: &na::Point<S, D>,
    b: &na::Point<S, D>,
) -> na::Point<S, D> {
    point_best(a, b, S::min)
}

fn point_max<S: 'static + Float + Debug, const D: usize>(
    a: &na::Point<S, D>,
    b: &na::Point<S, D>,
) -> na::Point<S, D> {
    point_best(a, b, S::max)
}

fn points_min<S: 'static + Float + Debug, const D: usize>(
    points: &[na::Point<S, D>],
) -> na::Point<S, D> {
    points_best(points, S::min)
}

fn points_max<S: 'static + Float + Debug, const D: usize>(
    points: &[na::Point<S, D>],
) -> na::Point<S, D> {
    points_best(points, S::max)
}

/// 3D Bounding Box - defined by two diagonally opposing points.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox<S: 'static + Debug + Copy + PartialEq, const D: usize> {
    /// X-Y-Z-Minimum corner of the box.
    pub min: na::Point<S, D>,
    /// X-Y-Z-Maximum corner of the box.
    pub max: na::Point<S, D>,
}

impl<S: Float + na::RealField, T: AsRef<[na::Point<S, D>]>, const D: usize> From<T>
    for BoundingBox<S, D>
{
    fn from(points: T) -> Self {
        Self {
            min: points_min(points.as_ref()),
            max: points_max(points.as_ref()),
        }
    }
}

impl<S: Float + Debug + na::RealField + simba::scalar::RealField, const D: usize>
    BoundingBox<S, D>
{
    /// Returns an infinte sized box.
    pub fn infinity() -> Self {
        Self {
            min: na::Point::from([S::neg_infinity(); D]),
            max: na::Point::from([S::infinity(); D]),
        }
    }
    /// Returns a negatively infinte sized box.
    pub fn neg_infinity() -> Self {
        Self {
            min: na::Point::from([S::infinity(); D]),
            max: na::Point::from([S::neg_infinity(); D]),
        }
    }
    /// Create a new Bounding Box by supplying two points.
    pub fn new(a: &na::Point<S, D>, b: &na::Point<S, D>) -> Self {
        Self {
            min: point_min(a, b),
            max: point_max(a, b),
        }
    }
    /// Returns true if the Bounding Box is empty.
    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }
    /// Returns true if the Bounding Box has finite size.
    pub fn is_finite(&self) -> bool {
        self.min
            .iter()
            .chain(self.max.iter())
            .all(|x| x.is_finite())
    }
    /// Returns the center point of the Bounding Box.
    pub fn center(&self) -> na::Point<S, D> {
        self.min + (self.max - self.min) / S::from(2.0).unwrap()
    }
    /// Create a CSG Union of two Bounding Boxes.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: point_min(&self.min, &other.min),
            max: point_max(&self.max, &other.max),
        }
    }
    /// Create a CSG Intersection of two Bounding Boxes.
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: point_max(&self.min, &other.min),
            max: point_min(&self.max, &other.max),
        }
    }
    /// Get the corners of the Bounding Box
    ///
    /// Warning: bounding box of dimension D has 2^D corners
    pub fn get_corners(&self) -> Vec<na::Point<S, 3>> {
        (0..2usize.pow(3))
            .map(|i| {
                na::Point::from(std::array::from_fn(|d| match (i >> d) & 1 {
                    0 => self.min[d],
                    1 => self.max[d],
                    _ => unreachable!(),
                }))
            })
            .collect()
    }
    /// Dilate a Bounding Box by some amount in all directions.
    pub fn dilate(&mut self, d: S) -> &mut Self {
        self.min.iter_mut().for_each(|coord| *coord -= d);
        self.max.iter_mut().for_each(|coord| *coord += d);
        self
    }
    /// Add a Point to a Bounding Box, e.g. expand the Bounding Box to contain that point.
    pub fn insert(&mut self, o: &na::Point<S, D>) -> &mut Self {
        self.min = point_min(&self.min, o);
        self.max = point_max(&self.max, o);
        self
    }
    /// Return the size of the Box.
    pub fn dim(&self) -> na::SVector<S, D> {
        self.max - self.min
    }
    /// Returns the approximate distance of p to the box. The result is guarateed to be not less
    /// than the euclidean distance of p to the box.
    pub fn distance(&self, point: &na::Point<S, D>) -> S {
        // If p is not inside (neg), then it is outside (pos) on only one side.
        // So so calculating the max of the diffs on both sides should result in the true value,
        // if positive.
        point_max(&(point - self.max).into(), &(self.min - point).into())
            .iter()
            .fold(S::neg_infinity(), |a, b| Float::max(a, *b))
    }
    /// Return true if the Bounding Box contains p.
    pub fn contains(&self, point: &na::Point<S, D>) -> bool {
        let is_bigger_than_min = self
            .min
            .iter()
            .zip(point.iter())
            .all(|(min_i, point_i)| *min_i <= *point_i);

        let is_smaller_than_max = self
            .max
            .iter()
            .zip(point.iter())
            .all(|(max_i, point_i)| *max_i >= *point_i);

        is_bigger_than_min && is_smaller_than_max
    }
}

impl<S: Float + Debug + na::RealField + simba::scalar::RealField> BoundingBox<S, 3> {
    /// Transform a Bounding Box - resulting in a enclosing axis aligned Bounding Box.
    pub fn transform(&self, mat: &na::Matrix4<S>) -> Self {
        let corners = self.get_corners();
        let transformed: Vec<_> = corners
            .into_iter()
            .map(|c| mat.transform_point(&c))
            .collect();
        Self::from(&transformed)
    }
}

impl<T: Float, const D: usize> AbsDiffEq for BoundingBox<T, D>
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

impl<T: Float, const D: usize> RelativeEq for BoundingBox<T, D>
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
        let bbox = BoundingBox::<f64, 3>::neg_infinity();
        assert!(bbox.is_empty());
        let bbox = BoundingBox::from([na::Point::from([0., 0., 0.])]);
        assert!(!bbox.is_empty());
    }

    #[test]
    fn is_finite() {
        let bbox = BoundingBox::<f64, 3>::neg_infinity();
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
