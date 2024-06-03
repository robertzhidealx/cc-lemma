  (declare-datatypes () ((MyBool (MyTrue) (MyFalse))))
  (declare-datatypes () ((Unit (Null))))
  (declare-datatypes () ((Nat (Zero) (Succ (proj_Succ_0 Nat)))))
  (declare-datatypes () ((List (Nil (proj_Nil_0 Unit)) (Cons (proj_Cons_0 Nat) (proj_Cons_1 List)))))
  (declare-datatypes () ((Zipper (Zip (proj_Zip_0 List) (proj_Zip_1 List)))))
  (declare-fun tf1 (List List) List)
  (declare-fun tf0 (List List) List)
  (declare-fun myconcat (List List) List)
  (declare-fun tf3 (List) List)
  (declare-fun tf2 (List) List)
  (declare-fun rev (List) List)
  (declare-fun plus (Nat Nat) Nat)
  (declare-fun tf5 (List) Nat)
  (declare-fun tf4 (List) Nat)
  (declare-fun sum (List) Nat)
  (declare-fun tf6 (Zipper) List)
  (declare-fun repr (Zipper) List)
  (declare-fun tf8 (Zipper) Zipper)
  (declare-fun tf7 (Zipper) Zipper)
  (declare-fun target (Zipper) Zipper)
  (declare-fun main (Zipper) Nat)
  (declare-fun tf10 (Zipper) Nat)
  (declare-fun tf9 (Zipper) Nat)
  (declare-fun targetNew (Zipper) Nat)
  (declare-fun mainNew (Zipper) Nat)
  (assert (forall ((tv6 Unit) (tv5 List)) (= (tf1 tv5 (Nil tv6)) tv5)))
  (assert (forall ((tv8 List) (tv7 Nat) (tv5 List)) (= (tf1 tv5 (Cons tv7 tv8)) (Cons tv7 (tf0 tv8 tv5)))))
  (assert (forall ((tv3 List) (tv2 List)) (= (tf0 tv2 tv3) (tf1 tv3 tv2))))
  (assert (forall ((tv1 List) (tv0 List)) (= (myconcat tv0 tv1) (tf0 tv0 tv1))))
  (assert (forall ((tv12 Unit)) (= (tf3 (Nil tv12)) (Nil Null))))
  (assert (forall ((tv14 List) (tv13 Nat)) (= (tf3 (Cons tv13 tv14)) (myconcat (tf2 tv14) (Cons tv13 (Nil Null))))))
  (assert (forall ((tv10 List)) (= (tf2 tv10) (tf3 tv10))))
  (assert (forall ((tv9 List)) (= (rev tv9) (tf2 tv9))))
  (assert (forall ((x Nat)) (= (plus Zero x) x)))
  (assert (forall ((y Nat) (x Nat)) (= (plus (Succ x) y) (Succ (plus x y)))))
  (assert (forall ((tv18 Unit)) (= (tf5 (Nil tv18)) Zero)))
  (assert (forall ((tv20 List) (tv19 Nat)) (= (tf5 (Cons tv19 tv20)) (plus tv19 (tf4 tv20)))))
  (assert (forall ((tv16 List)) (= (tf4 tv16) (tf5 tv16))))
  (assert (forall ((tv15 List)) (= (sum tv15) (tf4 tv15))))
  (assert (forall ((tv23 List) (tv22 List)) (= (tf6 (Zip tv22 tv23)) (myconcat (rev tv22) tv23))))
  (assert (forall ((tv21 Zipper)) (= (repr tv21) (tf6 tv21))))
  (assert (forall ((tv27 List) (tv26 List)) (= (tf8 (Zip tv26 tv27)) (Zip tv26 tv27))))
  (assert (forall ((tv25 Zipper)) (= (tf7 tv25) (tf8 tv25))))
  (assert (forall ((tv24 Zipper)) (= (target tv24) (tf7 tv24))))
  (assert (forall ((tv28 Zipper)) (= (main tv28) (sum (repr (target tv28))))))
  (assert (forall ((tv32 List) (tv31 List)) (= (tf10 (Zip tv31 tv32)) (plus (sum tv31) (sum tv32)))))
  (assert (forall ((tv30 Zipper)) (= (tf9 tv30) (tf10 tv30))))
  (assert (forall ((tv29 Zipper)) (= (targetNew tv29) (tf9 tv29))))
  (assert (forall ((tv33 Zipper)) (= (mainNew tv33) (targetNew tv33))))
  (assert (not (forall ((inp0 Zipper)) (= (main inp0) (mainNew inp0)))))
  (check-sat)
