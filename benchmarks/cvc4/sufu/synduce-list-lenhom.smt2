  (declare-datatypes () ((MyBool (MyTrue) (MyFalse))))
  (declare-datatypes () ((Unit (Null))))
  (declare-datatypes () ((Nat (Zero) (Succ (proj_Succ_0 Nat)))))
  (declare-datatypes () ((List (Nil (proj_Nil_0 Unit)) (Cons (proj_Cons_0 Nat) (proj_Cons_1 List)))))
  (declare-datatypes () ((CList (Cnil (proj_Cnil_0 Unit)) (Single (proj_Single_0 Nat)) (Concat (proj_Concat_0 CList) (proj_Concat_1 CList)))))
  (declare-fun plus (Nat Nat) Nat)
  (declare-fun tf1 (List) Nat)
  (declare-fun tf0 (List) Nat)
  (declare-fun spec (List) Nat)
  (declare-fun tf3 (List List) List)
  (declare-fun tf2 (List List) List)
  (declare-fun cat (List List) List)
  (declare-fun tf5 (CList) List)
  (declare-fun tf4 (CList) List)
  (declare-fun repr (CList) List)
  (declare-fun main (CList) Nat)
  (declare-fun tf7 (CList) Nat)
  (declare-fun tf6 (CList) Nat)
  (declare-fun reprNew (CList) Nat)
  (declare-fun mainNew (CList) Nat)
  (assert (forall ((x Nat)) (= (plus Zero x) x)))
  (assert (forall ((y Nat) (x Nat)) (= (plus (Succ x) y) (Succ (plus x y)))))
  (assert (forall ((tv3 Unit)) (= (tf1 (Nil tv3)) Zero)))
  (assert (forall ((tv5 List) (tv4 Nat)) (= (tf1 (Cons tv4 tv5)) (plus (Succ Zero) (tf0 tv5)))))
  (assert (forall ((tv1 List)) (= (tf0 tv1) (tf1 tv1))))
  (assert (forall ((tv0 List)) (= (spec tv0) (tf0 tv0))))
  (assert (forall ((tv12 Unit) (tv11 List)) (= (tf3 tv11 (Nil tv12)) tv11)))
  (assert (forall ((tv14 List) (tv13 Nat) (tv11 List)) (= (tf3 tv11 (Cons tv13 tv14)) (Cons tv13 (tf2 tv14 tv11)))))
  (assert (forall ((tv9 List) (tv8 List)) (= (tf2 tv8 tv9) (tf3 tv9 tv8))))
  (assert (forall ((tv7 List) (tv6 List)) (= (cat tv6 tv7) (tf2 tv6 tv7))))
  (assert (forall ((tv18 Unit)) (= (tf5 (Cnil tv18)) (Nil Null))))
  (assert (forall ((tv19 Nat)) (= (tf5 (Single tv19)) (Cons tv19 (Nil Null)))))
  (assert (forall ((tv21 CList) (tv20 CList)) (= (tf5 (Concat tv20 tv21)) (cat (tf4 tv20) (tf4 tv21)))))
  (assert (forall ((tv16 CList)) (= (tf4 tv16) (tf5 tv16))))
  (assert (forall ((tv15 CList)) (= (repr tv15) (tf4 tv15))))
  (assert (forall ((tv22 CList)) (= (main tv22) (spec (repr tv22)))))
  (assert (forall ((tv26 Unit)) (= (tf7 (Cnil tv26)) Zero)))
  (assert (forall ((tv27 Nat)) (= (tf7 (Single tv27)) (Succ Zero))))
  (assert (forall ((tv29 CList) (tv28 CList)) (= (tf7 (Concat tv28 tv29)) (plus (tf6 tv29) (tf6 tv28)))))
  (assert (forall ((tv24 CList)) (= (tf6 tv24) (tf7 tv24))))
  (assert (forall ((tv23 CList)) (= (reprNew tv23) (tf6 tv23))))
  (assert (forall ((tv30 CList)) (= (mainNew tv30) (reprNew tv30))))
  (assert (not (forall ((inp0 CList)) (= (main inp0) (mainNew inp0)))))
  (check-sat)
