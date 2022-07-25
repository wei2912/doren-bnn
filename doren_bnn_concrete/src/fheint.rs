use anyhow::Result;
use concrete::{prelude::*, ClientKey, DynInteger, DynShortInt};

use std::borrow::Borrow;
use std::fmt::{Debug, Error, Formatter};
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/* Define a new trait for primitive casting which might cause loss in precision. */
pub trait Cast<T> {
    fn cast(self) -> T;
}

impl Cast<f64> for u8 {
    fn cast(self) -> f64 {
        self as f64
    }
}

impl Cast<f64> for u64 {
    fn cast(self) -> f64 {
        self as f64
    }
}

pub trait FheIntPlaintext:
    From<u8> + Cast<f64> + Add<Output = Self> + AddAssign + Copy + Clone + Debug + Send
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

/* Wrapper struct for FHE unsigned integers with an f64 offset, added to the plaintext
 * upon decryption, and a max value.
 *
 * The offset and max value is used to maintain the invariant that the ciphertext
 * represents an integer in the range [0, max value].
 */
#[derive(Clone)]
pub struct FheInt<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>>(Option<U>, f64, T);

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> FheInt<T, U> {
    pub fn zero() -> Self {
        FheInt(None, 0.0, 0.into())
    }

    pub fn zero_with_offset(x: f64) -> Self {
        FheInt(None, x, 0.into())
    }

    pub fn encrypt_bin_pm<F: Fn(T) -> U>(enc_func: F, pt: bool) -> Self {
        let ct = enc_func(if pt { 2.into() } else { 0.into() });
        FheInt(Some(ct), -1.0, 2.into())
    }

    pub fn try_encrypt_bin_pm<E, F: Fn(T) -> Result<U, E>>(
        try_enc_func: F,
        pt: bool,
    ) -> Result<Self, E> {
        let ct = try_enc_func(if pt { 2.into() } else { 0.into() })?;
        Ok(FheInt(Some(ct), -1.0, 2.into()))
    }

    pub fn decrypt(&self, client_key: &ClientKey) -> f64 {
        let FheInt(opt, offset, _) = self;
        opt.as_ref()
            .map_or_else(|| 0.0, |ct| ct.decrypt(client_key).cast())
            + offset
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> Debug for FheInt<T, U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "(?, {:?}, {:?})", self.1, self.2)
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>, B> Add<B> for &FheInt<T, U>
where
    B: Borrow<FheInt<T, U>>,
{
    type Output = FheInt<T, U>;

    fn add(self, rhs: B) -> Self::Output {
        let rhs = rhs.borrow();
        FheInt(
            match (&self.0, &rhs.0) {
                (Some(x0), Some(y0)) => Some(x0.clone() + y0.clone()),
                (Some(x0), None) => Some(x0.clone()),
                (None, Some(y0)) => Some(y0.clone()),
                (None, None) => None,
            },
            self.1 + rhs.1,
            self.2 + rhs.2,
        )
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
        match &mut self.0 {
            Some(x0) => {
                if let Some(y0) = &rhs.0 {
                    *x0 += y0;
                }
            }
            None => {
                self.0 = rhs.0.clone();
            }
        }
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T>> Neg for &FheInt<T, U> {
    type Output = FheInt<T, U>;

    fn neg(self) -> Self::Output {
        FheInt(
            self.0.as_ref().map(|x0| (-x0.clone()) + self.2),
            -self.1 as f64 - self.2.cast(),
            self.2,
        )
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
    fn map<F: Fn(f64) -> u64>(&self, func: F, offset: f64, max_val: T) -> Self;
    fn apply<F: Fn(f64) -> u64>(&mut self, func: F, offset: f64, max_val: T);
}

impl<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T> + FheBootstrap> FheIntBootstrap<T, U>
    for FheInt<T, U>
{
    fn map<F: Fn(f64) -> u64>(&self, func: F, offset: f64, max_val: T) -> Self {
        match &self.0 {
            Some(ct) => FheInt(Some(ct.map(|pt| func(pt.cast() - self.1))), offset, max_val),
            None => FheInt(None, offset + func(self.1).cast(), max_val),
        }
    }

    fn apply<F: Fn(f64) -> u64>(&mut self, func: F, offset: f64, max_val: T) {
        match &mut self.0 {
            Some(ct) => {
                ct.apply(|pt| func(pt.cast() - self.1));
                self.1 = offset;
                self.2 = max_val;
            }
            None => {
                self.1 = offset + func(-self.1).cast();
                self.2 = max_val;
            }
        };
    }
}
