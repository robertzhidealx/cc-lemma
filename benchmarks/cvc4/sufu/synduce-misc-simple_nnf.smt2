  (declare-datatypes () ((MyBool (MyTrue) (MyFalse))))
  (declare-datatypes () ((Formula (Flit (proj_Flit_0 MyBool)) (Fand (proj_Fand_0 Formula) (proj_Fand_1 Formula)) (Forr (proj_Forr_0 Formula) (proj_Forr_1 Formula)) (Fnot (proj_Fnot_0 Formula)))))
  (declare-datatypes () ((NnfFormula (Nfneglit (proj_Nfneglit_0 MyBool)) (Nflit (proj_Nflit_0 MyBool)) (Nfand (proj_Nfand_0 NnfFormula) (proj_Nfand_1 NnfFormula)) (Nfor (proj_Nfor_0 NnfFormula) (proj_Nfor_1 NnfFormula)))))
  (declare-fun myand (MyBool MyBool) MyBool)
  (declare-fun myor (MyBool MyBool) MyBool)
  (declare-fun ite2 (MyBool) MyBool)
  (declare-fun tf1 (Formula) MyBool)
  (declare-fun tf0 (Formula) MyBool)
  (declare-fun spec (Formula) MyBool)
  (declare-fun tf3 (NnfFormula) Formula)
  (declare-fun tf2 (NnfFormula) Formula)
  (declare-fun repr (NnfFormula) Formula)
  (declare-fun main (NnfFormula) MyBool)
  (declare-fun mynot (MyBool) MyBool)
  (declare-fun tf5 (NnfFormula) MyBool)
  (declare-fun tf4 (NnfFormula) MyBool)
  (declare-fun reprNew (NnfFormula) MyBool)
  (declare-fun mainNew (NnfFormula) MyBool)
  (assert (forall ((x MyBool)) (= (myand MyFalse x) MyFalse)))
  (assert (forall ((true MyBool)) (= (myand true MyFalse) MyFalse)))
  (assert (= (myand MyTrue MyTrue) MyTrue))
  (assert (forall ((x MyBool)) (= (myor MyTrue x) MyTrue)))
  (assert (forall ((false MyBool)) (= (myor false MyTrue) MyTrue)))
  (assert (= (myor MyFalse MyFalse) MyFalse))
  (assert (= (ite2 MyTrue) MyFalse))
  (assert (= (ite2 MyFalse) MyTrue))
  (assert (forall ((tv3 MyBool)) (= (tf1 (Flit tv3)) tv3)))
  (assert (forall ((tv5 Formula) (tv4 Formula)) (= (tf1 (Fand tv4 tv5)) (myand (tf0 tv4) (tf0 tv5)))))
  (assert (forall ((tv7 Formula) (tv6 Formula)) (= (tf1 (Forr tv6 tv7)) (myor (tf0 tv6) (tf0 tv7)))))
  (assert (forall ((tv8 Formula)) (= (tf1 (Fnot tv8)) (ite2 (tf0 tv8)))))
  (assert (forall ((tv1 Formula)) (= (tf0 tv1) (tf1 tv1))))
  (assert (forall ((tv0 Formula)) (= (spec tv0) (tf0 tv0))))
  (assert (forall ((tv12 MyBool)) (= (tf3 (Nflit tv12)) (Flit tv12))))
  (assert (forall ((tv13 MyBool)) (= (tf3 (Nfneglit tv13)) (Fnot (Flit tv13)))))
  (assert (forall ((tv15 NnfFormula) (tv14 NnfFormula)) (= (tf3 (Nfand tv14 tv15)) (Fand (tf2 tv14) (tf2 tv15)))))
  (assert (forall ((tv17 NnfFormula) (tv16 NnfFormula)) (= (tf3 (Nfor tv16 tv17)) (Forr (tf2 tv16) (tf2 tv17)))))
  (assert (forall ((tv10 NnfFormula)) (= (tf2 tv10) (tf3 tv10))))
  (assert (forall ((tv9 NnfFormula)) (= (repr tv9) (tf2 tv9))))
  (assert (forall ((tv18 NnfFormula)) (= (main tv18) (spec (repr tv18)))))
  (assert (= (mynot MyTrue) MyFalse))
  (assert (= (mynot MyFalse) MyTrue))
  (assert (forall ((tv22 MyBool)) (= (tf5 (Nflit tv22)) tv22)))
  (assert (forall ((tv23 MyBool)) (= (tf5 (Nfneglit tv23)) (mynot tv23))))
  (assert (forall ((tv25 NnfFormula) (tv24 NnfFormula)) (= (tf5 (Nfand tv24 tv25)) (myand (tf4 tv25) (tf4 tv24)))))
  (assert (forall ((tv27 NnfFormula) (tv26 NnfFormula)) (= (tf5 (Nfor tv26 tv27)) (myor (tf4 tv26) (tf4 tv27)))))
  (assert (forall ((tv20 NnfFormula)) (= (tf4 tv20) (tf5 tv20))))
  (assert (forall ((tv19 NnfFormula)) (= (reprNew tv19) (tf4 tv19))))
  (assert (forall ((tv28 NnfFormula)) (= (mainNew tv28) (reprNew tv28))))
  (assert (not (forall ((inp0 NnfFormula)) (= (main inp0) (mainNew inp0)))))
  (check-sat)
