use anyhow::Result;
use concrete::{prelude::*, ClientKey, DynInteger, DynShortInt};

use std::borrow::Borrow;
use std::fmt::{Debug, Error, Formatter};
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

fn unimplemented_add_diff_weights() {
    unimplemented!("addition of ciphertexts with different non-zero weights is not supported");
}

/* Trait for primitive casting which might cause loss in precision. */
pub trait CastInto<T> {
    fn cast_into(self) -> T;
}

impl CastInto<f64> for u8 {
    fn cast_into(self) -> f64 {
        self as f64
    }
}

impl CastInto<f64> for u64 {
    fn cast_into(self) -> f64 {
        self as f64
    }
}

/* Trait for primitive casting which may be fallible with overflow. */
pub trait CastFrom<T>: Sized {
    fn cast_from(_: T) -> Self;
}

impl CastFrom<u64> for u8 {
    fn cast_from(x: u64) -> u8 {
        if x > u8::MAX.into() {
            panic!("x is not in the range of [0, u8::MAX]");
        }
        x as u8
    }
}

impl CastFrom<u64> for u64 {
    fn cast_from(x: u64) -> u64 {
        x
    }
}

pub trait FheIntPlaintext:
    CastFrom<u64>
    + Into<u64>
    + CastInto<f64>
    + Add<Output = Self>
    + AddAssign
    + Copy
    + Clone
    + Debug
    + Send
{
}

impl FheIntPlaintext for u8 {}
impl FheIntPlaintext for u64 {}

pub trait FheIntCiphertext<'a, T: FheIntPlaintext>:
    'a
    + FheDecrypt<T>
    + Add<Output = Self>
    + Add<T, Output = Self>
    + AddAssign<Self>
    + AddAssign<&'a Self>
    + Neg<Output = Self>
    + Sub<Output = Self>
    + Sub<T, Output = Self>
    + SubAssign
    + Clone
    + Sized
    + Send
{
}

impl FheIntCiphertext<'_, u8> for DynShortInt {}
impl FheIntCiphertext<'_, u64> for DynInteger {}

/* Wrapper struct for FHE unsigned integers with a f64 weight and bias, used to
 * transform the plaintext upon decryption, and a max value.
 *
 * The ciphertext should always encrypt an integer in the range [0, max_value].
 * TODO: Error checking to ensure above invariant holds.
 */
#[derive(Clone)]
pub struct FheInt<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> {
    ct_opt: Option<U>,
    weight: f64,
    bias: f64,
    max_pt: T,
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> FheInt<T, U> {
    pub fn zero() -> Self {
        FheInt {
            ct_opt: None,
            weight: 0.0,
            bias: 0.0,
            max_pt: T::cast_from(0),
        }
    }

    pub fn encrypt_bin_pm<F: Fn(T) -> U>(enc_func: F, pt: bool) -> Self {
        let ct = enc_func(T::cast_from(pt as u64));
        FheInt {
            ct_opt: Some(ct),
            weight: 2.0,
            bias: -1.0,
            max_pt: T::cast_from(1),
        }
    }

    pub fn try_encrypt_bin_pm<E: std::fmt::Debug, F: Fn(T) -> Result<U, E>>(
        try_enc_func: F,
        pt: bool,
    ) -> Result<Self, E> {
        let ct = try_enc_func(T::cast_from(pt as u64))?;
        Ok(FheInt {
            ct_opt: Some(ct),
            weight: 2.0,
            bias: -1.0,
            max_pt: T::cast_from(1),
        })
    }

    pub fn decrypt(&self, client_key: &ClientKey) -> f64 {
        let FheInt {
            ct_opt,
            weight,
            bias,
            ..
        } = self;
        ct_opt
            .to_owned()
            .map_or_else(|| *bias, |ct| weight * ct.decrypt(client_key).cast_into() + bias)
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> Debug for FheInt<T, U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "{:?} * [0, {:?}] + {:?}",
            self.weight, self.max_pt, self.bias,
        )
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>, B> Add<B> for &FheInt<T, U>
where
    B: Borrow<FheInt<T, U>>,
{
    type Output = FheInt<T, U>;

    fn add(self, rhs: B) -> Self::Output {
        let rhs = rhs.borrow();
        if self.weight != 0.0 && rhs.weight != 0.0 && self.weight != rhs.weight {
            unimplemented_add_diff_weights();
        }

        FheInt {
            ct_opt: match (&self.ct_opt, &rhs.ct_opt) {
                (Some(ct0), Some(ct1)) => Some(ct0.clone() + ct1.clone()),
                (Some(ct0), None) => Some(ct0.clone()),
                (None, Some(ct1)) => Some(ct1.clone()),
                (None, None) => None,
            },
            weight: f64::max(self.weight, rhs.weight),
            bias: self.bias + rhs.bias,
            max_pt: self.max_pt + rhs.max_pt,
        }
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>, B> Add<B> for FheInt<T, U>
where
    B: Borrow<Self>,
{
    type Output = Self;

    fn add(self, rhs: B) -> Self::Output {
        &self + rhs
    }
}

impl<
        T: FheIntPlaintext,
        U: for<'a> FheIntCiphertext<'a, T> + for<'a> AddAssign<&'a U>,
        B: Borrow<Self>,
    > AddAssign<B> for FheInt<T, U>
{
    fn add_assign(&mut self, rhs: B) {
        let rhs = rhs.borrow();
        if self.weight != 0.0 && rhs.weight != 0.0 && self.weight != rhs.weight {
            unimplemented_add_diff_weights();
        }

        match self.ct_opt.as_mut() {
            Some(ct0) => match &rhs.ct_opt {
                Some(ct1) => {
                    *ct0 += ct1;
                }
                None => {}
            },
            None => {
                self.ct_opt = rhs.ct_opt.clone();
            }
        }
        self.weight = f64::max(self.weight, rhs.weight);
        self.bias += rhs.bias;
        self.max_pt += rhs.max_pt;
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> Neg for &FheInt<T, U> {
    type Output = FheInt<T, U>;

    fn neg(self) -> Self::Output {
        FheInt {
            ct_opt: self
                .ct_opt
                .as_ref()
                .map(|ct| (-ct.clone()) + self.max_pt.clone()),
            weight: self.weight,
            bias: -self.bias - self.weight * self.max_pt.cast_into(),
            max_pt: self.max_pt,
        }
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> Neg for FheInt<T, U> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>, B> Sub<B> for &FheInt<T, U>
where
    B: Borrow<FheInt<T, U>>,
{
    type Output = FheInt<T, U>;

    fn sub(self, rhs: B) -> Self::Output {
        self + (-rhs.borrow())
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>, B> Sub<B> for FheInt<T, U>
where
    B: Borrow<Self>,
{
    type Output = Self;

    fn sub(self, rhs: B) -> Self::Output {
        &self - rhs
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T> + for<'a> AddAssign<&'a U>, B>
    SubAssign<B> for FheInt<T, U>
where
    B: Borrow<FheInt<T, U>>,
{
    fn sub_assign(&mut self, rhs: B) {
        *self += -rhs.borrow();
    }
}

/* Wrapper trait for FheBootstrap on FheInt, to enable bootstrapping into restricted
 * domains beyond u64.
 */
pub trait FheIntBootstrap<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T> + FheBootstrap>:
    Clone + Send
{
    fn map<F: Fn(f64) -> u64>(&self, func: F, weight: f64, bias: f64) -> Self;
    fn apply<F: Fn(f64) -> u64>(&mut self, func: F, weight: f64, bias: f64);
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T> + FheBootstrap> FheIntBootstrap<T, U>
    for FheInt<T, U>
{
    fn map<F: Fn(f64) -> u64>(&self, func: F, weight: f64, bias: f64) -> Self {
        let f = |pt: u64| -> u64 { func(self.weight * pt.cast_into() + self.bias) };

        let f_pt_range = (0..self.max_pt.into() + 1).map(f);
        let f_pt_max = f_pt_range.max().unwrap();

        match &self.ct_opt {
            Some(ct) => FheInt {
                ct_opt: Some(ct.map(f)),
                weight: weight,
                bias: bias,
                max_pt: T::cast_from(f_pt_max),
            },
            None => FheInt {
                ct_opt: None,
                weight: 0.0,
                bias: weight * f(0).cast_into() + bias,
                max_pt: T::cast_from(0),
            },
        }
    }

    fn apply<F: Fn(f64) -> u64>(&mut self, func: F, weight: f64, bias: f64) {
        let f = |pt: u64| -> u64 { func(self.weight * pt.cast_into() + self.bias) };

        let f_pt_range = (0..self.max_pt.into() + 1).map(f);
        let f_pt_max = f_pt_range.max().unwrap();

        match &mut self.ct_opt {
            Some(ct) => {
                ct.apply(f);
                self.weight = weight;
                self.bias = bias;
                self.max_pt = T::cast_from(f_pt_max);
            }
            None => {
                self.bias = weight * f(0).cast_into() + bias;
                self.weight = 0.0;
                self.max_pt = T::cast_from(0);
            }
        };
    }
}
