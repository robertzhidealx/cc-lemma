  (declare-datatypes () ((MyBool (MyTrue) (MyFalse))))
  (declare-datatypes () ((Unit (Null))))
  (declare-datatypes () ((Nat (Zero) (Succ (proj_Succ_0 Nat)))))
  (declare-datatypes () ((CList (Cnil (proj_Cnil_0 Unit)) (Single (proj_Single_0 Nat)) (Concat (proj_Concat_0 CList) (proj_Concat_1 CList)))))
  (declare-datatypes () ((List (Nil (proj_Nil_0 Unit)) (Cons (proj_Cons_0 Nat) (proj_Cons_1 List)))))
  (declare-fun tf1 (List List) List)
  (declare-fun tf0 (List List) List)
  (declare-fun cat (List List) List)
  (declare-fun tf3 (CList) List)
  (declare-fun tf2 (CList) List)
  (declare-fun repr (CList) List)
  (declare-fun plus (Nat Nat) Nat)
  (declare-fun tf5 (List) Nat)
  (declare-fun tf4 (List) Nat)
  (declare-fun spec (List) Nat)
  (declare-fun main (CList) Nat)
  (declare-fun tf7 (CList) Nat)
  (declare-fun tf6 (CList) Nat)
  (declare-fun reprNew (CList) Nat)
  (declare-fun mainNew (CList) Nat)
  (assert (forall ((tv7 List) (tv6 Nat) (tv5 List)) (= (tf1 tv5 (Cons tv6 tv7)) (Cons tv6 (tf0 tv7 tv5)))))
  (assert (forall ((tv8 Unit) (tv5 List)) (= (tf1 tv5 (Nil tv8)) tv5)))
  (assert (forall ((tv3 List) (tv2 List)) (= (tf0 tv2 tv3) (tf1 tv3 tv2))))
  (assert (forall ((tv1 List) (tv0 List)) (= (cat tv0 tv1) (tf0 tv0 tv1))))
  (assert (forall ((tv12 Unit)) (= (tf3 (Cnil tv12)) (Nil Null))))
  (assert (forall ((tv13 Nat)) (= (tf3 (Single tv13)) (Cons tv13 (Nil Null)))))
  (assert (forall ((tv15 CList) (tv14 CList)) (= (tf3 (Concat tv14 tv15)) (cat (tf2 tv14) (tf2 tv15)))))
  (assert (forall ((tv10 CList)) (= (tf2 tv10) (tf3 tv10))))
  (assert (forall ((tv9 CList)) (= (repr tv9) (tf2 tv9))))
  (assert (forall ((x Nat)) (= (plus Zero x) x)))
  (assert (forall ((y Nat) (x Nat)) (= (plus (Succ x) y) (Succ (plus x y)))))
  (assert (forall ((tv19 Unit)) (= (tf5 (Nil tv19)) Zero)))
  (assert (forall ((tv21 List) (tv20 Nat)) (= (tf5 (Cons tv20 tv21)) (plus tv20 (tf4 tv21)))))
  (assert (forall ((tv17 List)) (= (tf4 tv17) (tf5 tv17))))
  (assert (forall ((tv16 List)) (= (spec tv16) (tf4 tv16))))
  (assert (forall ((tv22 CList)) (= (main tv22) (spec (repr tv22)))))
  (assert (forall ((tv26 Unit)) (= (tf7 (Cnil tv26)) Zero)))
  (assert (forall ((tv27 Nat)) (= (tf7 (Single tv27)) tv27)))
  (assert (forall ((tv29 CList) (tv28 CList)) (= (tf7 (Concat tv28 tv29)) (plus (tf6 tv28) (tf6 tv29)))))
  (assert (forall ((tv24 CList)) (= (tf6 tv24) (tf7 tv24))))
  (assert (forall ((tv23 CList)) (= (reprNew tv23) (tf6 tv23))))
  (assert (forall ((tv30 CList)) (= (mainNew tv30) (reprNew tv30))))
  (assert (not (forall ((inp0 CList)) (= (main inp0) (mainNew inp0)))))
  (check-sat)
