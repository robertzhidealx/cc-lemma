  (declare-datatypes () ((MyBool (MyTrue) (MyFalse))))
  (declare-datatypes () ((Unit (Null))))
  (declare-datatypes () ((List (Cons (proj_Cons_0 MyBool) (proj_Cons_1 List)) (Nil (proj_Nil_0 Unit)))))
  (declare-fun tf1 (List List) List)
  (declare-fun tf0 (List) List)
  (declare-datatypes () ((Nat (Zero) (Succ (proj_Succ_0 Nat)))))
  (declare-fun tf2 (List) Nat)
  (declare-fun singlepass (List) Nat)
  (declare-fun myand (MyBool MyBool) MyBool)
  (declare-fun ite1 (MyBool Nat Nat) Nat)
  (declare-fun mynot (MyBool) MyBool)
  (declare-fun myor (MyBool MyBool) MyBool)
  (declare-fun plus (Nat Nat) Nat)
  (declare-fun tf4 (MyBool MyBool List) Nat)
  (declare-fun tf3 (MyBool MyBool List) Nat)
  (declare-fun count10p (List) Nat)
  (declare-fun main (List) Nat)
  (declare-fun tf5 (List) MyBool)
  (declare-fun alhead (List) MyBool)
  (declare-datatypes () ((Tuple2 (MakeTuple2 (proj_MakeTuple2_0 Nat) (proj_MakeTuple2_1 Nat)))))
  (declare-fun nateq (Nat Nat) MyBool)
  (declare-fun fst2 (Tuple2) Nat)
  (declare-fun snd2 (Tuple2) Nat)
  (declare-fun tf7 (List) Tuple2)
  (declare-fun tf6 (List) Tuple2)
  (declare-fun tf8 (List) Nat)
  (declare-fun singlepassNew (List) Nat)
  (declare-fun mainNew (List) Nat)
  (assert (forall ((tv5 Unit) (tv4 List)) (= (tf1 tv4 (Nil tv5)) tv4)))
  (assert (forall ((tv7 List) (tv6 MyBool) (tv4 List)) (= (tf1 tv4 (Cons tv6 tv7)) (Cons tv6 (tf0 tv7)))))
  (assert (forall ((tv2 List)) (= (tf0 tv2) (tf1 tv2 tv2))))
  (assert (forall ((tv9 List)) (= (tf2 tv9) (count10p (tf0 tv9)))))
  (assert (forall ((tv1 List)) (= (singlepass tv1) (tf2 tv1))))
  (assert (forall ((x MyBool)) (= (myand MyFalse x) MyFalse)))
  (assert (forall ((true MyBool)) (= (myand true MyFalse) MyFalse)))
  (assert (= (myand MyTrue MyTrue) MyTrue))
  (assert (forall ((y Nat) (x Nat)) (= (ite1 MyTrue x y) x)))
  (assert (forall ((y Nat) (x Nat)) (= (ite1 MyFalse x y) y)))
  (assert (= (mynot MyTrue) MyFalse))
  (assert (= (mynot MyFalse) MyTrue))
  (assert (forall ((x MyBool)) (= (myor MyTrue x) MyTrue)))
  (assert (forall ((false MyBool)) (= (myor false MyTrue) MyTrue)))
  (assert (= (myor MyFalse MyFalse) MyFalse))
  (assert (forall ((x Nat)) (= (plus Zero x) x)))
  (assert (forall ((y Nat) (x Nat)) (= (plus (Succ x) y) (Succ (plus x y)))))
  (assert (forall ((tv16 MyBool) (tv17 Unit) (tv15 MyBool)) (= (tf4 tv15 tv16 (Nil tv17)) Zero)))
  (assert (forall ((tv16 MyBool) (tv19 List) (tv18 MyBool) (tv15 MyBool)) (= (tf4 tv15 tv16 (Cons tv18 tv19)) (plus (ite1 (myand tv16 tv18) (Succ Zero) Zero) (tf3 tv18 (myand (mynot tv18) (myor tv15 tv16)) tv19)))))
  (assert (forall ((tv13 List) (tv12 MyBool) (tv11 MyBool)) (= (tf3 tv11 tv12 tv13) (tf4 tv11 tv12 tv13))))
  (assert (forall ((tv10 List)) (= (count10p tv10) (tf3 MyFalse MyFalse tv10))))
  (assert (forall ((tv20 List)) (= (main tv20) (singlepass tv20))))
  (assert (forall ((tv22 Unit)) (= (tf5 (Nil tv22)) MyFalse)))
  (assert (forall ((tv24 List) (tv23 MyBool)) (= (tf5 (Cons tv23 tv24)) tv23)))
  (assert (forall ((tv21 List)) (= (alhead tv21) (tf5 tv21))))
  (assert (= (nateq Zero Zero) MyTrue))
  (assert (forall ((x Nat)) (= (nateq Zero (Succ x)) MyFalse)))
  (assert (forall ((x Nat)) (= (nateq (Succ x) Zero) MyFalse)))
  (assert (forall ((y Nat) (x Nat)) (= (nateq (Succ x) (Succ y)) (nateq x y))))
  (assert (forall ((x1 Nat) (x0 Nat)) (= (fst2 (MakeTuple2 x0 x1)) x0)))
  (assert (forall ((x1 Nat) (x0 Nat)) (= (snd2 (MakeTuple2 x0 x1)) x1)))
  (assert (forall ((tv29 Unit)) (= (tf7 (Nil tv29)) (MakeTuple2 Zero Zero))))
  (assert (forall ((tv31 List) (tv30 MyBool)) (= (tf7 (Cons tv30 tv31)) (MakeTuple2 (ite1 (myor (myor (mynot tv30) (nateq (fst2 (tf6 tv31)) (snd2 (tf6 tv31)))) (alhead tv31)) (fst2 (tf6 tv31)) (plus (Succ Zero) (fst2 (tf6 tv31)))) (ite1 (mynot tv30) (snd2 (tf6 tv31)) (plus (Succ Zero) (snd2 (tf6 tv31))))))))
  (assert (forall ((tv27 List)) (= (tf6 tv27) (tf7 tv27))))
  (assert (forall ((tv32 List)) (= (tf8 tv32) (fst2 (tf6 tv32)))))
  (assert (forall ((tv26 List)) (= (singlepassNew tv26) (tf8 tv26))))
  (assert (forall ((tv33 List)) (= (mainNew tv33) (singlepassNew tv33))))
  (assert (not (forall ((inp0 List)) (= (main inp0) (mainNew inp0)))))
  (check-sat)
