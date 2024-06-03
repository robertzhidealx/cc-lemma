  (declare-datatypes () ((MyBool (MyTrue) (MyFalse))))
  (declare-datatypes () ((Unit (Null))))
  (declare-datatypes () ((List (Cons (proj_Cons_0 MyBool) (proj_Cons_1 List)) (Nil (proj_Nil_0 Unit)))))
  (declare-fun tf1 (List List) List)
  (declare-fun tf0 (List) List)
  (declare-datatypes () ((Nat (Zero) (Succ (proj_Succ_0 Nat)))))
  (declare-fun tf2 (List) Nat)
  (declare-fun singlepass (List) Nat)
  (declare-fun lq (Nat Nat) MyBool)
  (declare-fun ite1 (MyBool Nat Nat) Nat)
  (declare-fun max (Nat Nat) Nat)
  (declare-fun mynot (MyBool) MyBool)
  (declare-fun plus (Nat Nat) Nat)
  (declare-fun tf4 (Nat List) Nat)
  (declare-fun tf3 (Nat List) Nat)
  (declare-fun maxdistbetweenzeros (List) Nat)
  (declare-fun main (List) Nat)
  (declare-datatypes () ((Tuple2 (MakeTuple2 (proj_MakeTuple2_0 Nat) (proj_MakeTuple2_1 Nat)))))
  (declare-fun myor (MyBool MyBool) MyBool)
  (declare-fun snd2 (Tuple2) Nat)
  (declare-fun fst2 (Tuple2) Nat)
  (declare-fun tf6 (List) Tuple2)
  (declare-fun tf5 (List) Tuple2)
  (declare-fun tf7 (List) Nat)
  (declare-fun singlepassNew (List) Nat)
  (declare-fun mainNew (List) Nat)
  (assert (forall ((tv5 Unit) (tv4 List)) (= (tf1 tv4 (Nil tv5)) tv4)))
  (assert (forall ((tv7 List) (tv6 MyBool) (tv4 List)) (= (tf1 tv4 (Cons tv6 tv7)) (Cons tv6 (tf0 tv7)))))
  (assert (forall ((tv2 List)) (= (tf0 tv2) (tf1 tv2 tv2))))
  (assert (forall ((tv9 List)) (= (tf2 tv9) (maxdistbetweenzeros (tf0 tv9)))))
  (assert (forall ((tv1 List)) (= (singlepass tv1) (tf2 tv1))))
  (assert (= (lq Zero Zero) MyFalse))
  (assert (forall ((x Nat)) (= (lq Zero (Succ x)) MyTrue)))
  (assert (forall ((x Nat)) (= (lq (Succ x) Zero) MyFalse)))
  (assert (forall ((y Nat) (x Nat)) (= (lq (Succ x) (Succ y)) (lq x y))))
  (assert (forall ((y Nat) (x Nat)) (= (ite1 MyTrue x y) x)))
  (assert (forall ((y Nat) (x Nat)) (= (ite1 MyFalse x y) y)))
  (assert (forall ((tv11 Nat) (tv10 Nat)) (= (max tv10 tv11) (ite1 (lq tv10 tv11) tv11 tv10))))
  (assert (= (mynot MyTrue) MyFalse))
  (assert (= (mynot MyFalse) MyTrue))
  (assert (forall ((x Nat)) (= (plus Zero x) x)))
  (assert (forall ((y Nat) (x Nat)) (= (plus (Succ x) y) (Succ (plus x y)))))
  (assert (forall ((tv17 Unit) (tv16 Nat)) (= (tf4 tv16 (Nil tv17)) Zero)))
  (assert (forall ((tv19 List) (tv18 MyBool) (tv16 Nat)) (= (tf4 tv16 (Cons tv18 tv19)) (max (ite1 (mynot tv18) Zero (plus tv16 (Succ Zero))) (tf3 (ite1 (mynot tv18) Zero (plus tv16 (Succ Zero))) tv19)))))
  (assert (forall ((tv14 List) (tv13 Nat)) (= (tf3 tv13 tv14) (tf4 tv13 tv14))))
  (assert (forall ((tv12 List)) (= (maxdistbetweenzeros tv12) (tf3 Zero tv12))))
  (assert (forall ((tv20 List)) (= (main tv20) (singlepass tv20))))
  (assert (forall ((x MyBool)) (= (myor MyTrue x) MyTrue)))
  (assert (forall ((false MyBool)) (= (myor false MyTrue) MyTrue)))
  (assert (= (myor MyFalse MyFalse) MyFalse))
  (assert (forall ((x1 Nat) (x0 Nat)) (= (snd2 (MakeTuple2 x0 x1)) x1)))
  (assert (forall ((x1 Nat) (x0 Nat)) (= (fst2 (MakeTuple2 x0 x1)) x0)))
  (assert (forall ((tv25 Unit)) (= (tf6 (Nil tv25)) (MakeTuple2 Zero Zero))))
  (assert (forall ((tv27 List) (tv26 MyBool)) (= (tf6 (Cons tv26 tv27)) (MakeTuple2 (ite1 (myor (mynot tv26) (lq (snd2 (tf5 tv27)) (fst2 (tf5 tv27)))) (fst2 (tf5 tv27)) (plus (Succ Zero) (fst2 (tf5 tv27)))) (ite1 (mynot tv26) Zero (plus (Succ Zero) (snd2 (tf5 tv27))))))))
  (assert (forall ((tv23 List)) (= (tf5 tv23) (tf6 tv23))))
  (assert (forall ((tv28 List)) (= (tf7 tv28) (fst2 (tf5 tv28)))))
  (assert (forall ((tv22 List)) (= (singlepassNew tv22) (tf7 tv22))))
  (assert (forall ((tv29 List)) (= (mainNew tv29) (singlepassNew tv29))))
  (assert (not (forall ((inp0 List)) (= (main inp0) (mainNew inp0)))))
  (check-sat)
