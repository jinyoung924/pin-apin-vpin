/* ********************************************************************************* */
/* ************** W R D S   R E S E A R C H   A P P L I C A T I O N S ************** */
/* ********************************************************************************* */
/* Program   : PIN Estimation - WRDS.sas */
/* Summary   : Calculate the Easley, O'Hara, et al., PIN measure */
/* Date      : 2001, Modified May 2011, Expanded on Oct 2019 */
/* Author    : Premal P. Vora, Penn State; JinGi HA, AUT in New Zealand; Noah Stoffman (http://kelley.iu.edu/nstoffma) */
/* Details   : Program receives as input a file containing the count of buy orders */
/*             and sell orders for time periods t=1,T. The file is sorted by */
/*             symbol. The program then finds the probability of informed trading */
/*             for each symbol using a maximum likelihood estimation method. */
/* Modification: Compute PIN in 3 x 3 x 3 x 3 x 3 grids of Alpha, Delta, Mu, EtaB, and EtaS */
/*                     to reduce the chance of covergence failture in PROC NLMIXED. */
/* ********************************************************************************* */
 
/*--------------------------------------------------------------------*/
/*    Create an example dataset containing the count of buy orders    */
/*    and sell orders. For SYMBOL XYZ, the observations are           */
/*    consistent with the Easley, Hvidjkaer, and O'Hara example on    */
/*    p. 2198 of Easley, David, Soeren Hvidkjaer, and Maureen O'Hara, */
/*    2002, "Is Information Risk a Determinant of Asset Returns?",    */
/*    The Journal of Finance, 57 (5), pp. 2185-2221.                  */
/*--------------------------------------------------------------------*/

/*************** EDIT HERE ***************/
libname input "C:\TempSD";
libname output "D:\";

%let COUNTRY=US;
/****************************************/

/*--------------------------------------------------------------------*/
/*    Get the data ready for the NLMXIED procedure.                       */
/*--------------------------------------------------------------------*/
data buysell; set input.&COUNTRY._TRADE; 
Month=intnx("Month", Date, 0, 'b');
format Month yymmddn8.;
keep Symbol Date Month B S; run;
data Month; set buysell;
keep Symbol Month; run;

proc sort data=buysell; by Symbol Month Date; 
proc sort data=Month nodup; by Symbol Month; quit;
proc sql;
create table _estdat as
	select distinct a.Symbol, a.Month, c.Date as RegDate, c.B, c.S
	from Month as a, buysell as c
	where a.Symbol=c.Symbol and -2<=intck("Month", a.Month, c.Date)<=0
	group by a.Symbol, a.Month having count(c.Date)>20;
quit;

data _estdat; set _estdat;
where Month<'01AUG2020'd;
	run;

/*--------------------------------------------------------------------*/
/*    This is the procedure used to maximize the log likelihood       */
/*    function specified below.                                       */
/*                                                                    */
/*    The parameters emerging from this estimation, for SYMBOL XYZ    */
/*    are consistent with what's mentioned in the Easley, Hvidjkaer,  */
/*    O'Hara paper.                                                   */
/*--------------------------------------------------------------------*/

proc sort data=_estdat; by Symbol; quit;
data firm; set _estdat; keep Symbol; 
proc sort data=firm nodup; by Symbol; run;
%let NOBS=&SYSNOBS; * Do not move this line. It should follow the database of 'firm'.;

%macro APIN;
option nonotes;
%do FPRD=1 %to &NOBS;
%put &FPRD / &NOBS Processed...;

data target; set firm; if _n_=&FPRD; run;
data estdat; merge _estdat (in=a) target (in=t); if a and t; by Symbol; run;

proc printto log=junk2; run;
proc nlmixed data=estdat fd=central technique=quanew update=bfgs;
by Symbol Month;
	ods output ParameterEstimates=pinest1;
	parms a=.1 .5 .9, d=.1 .5 .9, t1=.1 .5 .9, t2=.1 .5 .9, 
				ub=20 200 2000, us=20 200 2000, 
				eb=20 200 2000, es=20 200 2000,
				pb=20 200 2000, ps=20 200 2000; 
	bounds 0 <= a d t1 t2 <= 1, ub us eb es pb ps >= 0;

	temp = (1-a)*(1-t1)*pdf('poisson',B,eb)*pdf('poisson',S,es)
				+ (1-a)*t1*pdf('poisson',B,eb+pb)*pdf('poisson',S,es+ps)
				+ a*(1-t2)*(1-d)*pdf('poisson',B,eb)*pdf('poisson',S,us+es)
				+ a*t2*(1-d)*pdf('poisson',B,eb+pb)*pdf('poisson',S,us+es+ps)
				+ a*(1-t2)*d*pdf('poisson',B,ub+eb)*pdf('poisson',S,es)
				+ a*t1*d*pdf('poisson',B,ub+eb+pb)*pdf('poisson',S,es+ps);

	if temp = 0 then temp = 1E-300;
	loglik = log(temp);

	model B~general(loglik);

	estimate 'alpha' a;
	estimate 'delta' d;
	estimate 'theta1' t1;
	estimate 'theta2' t2;
	estimate 'mub' ub;
	estimate 'mus' us;
	estimate 'epsilonb' eb;
	estimate 'epsilons' es;
	estimate 'pb' pb;
	estimate 'ps' ps;
quit;;
proc printto; run;
/*dm 'odsresults; clear';*/

/*--------------------------------------------------------------------*/
/*    Clean the output and print it nicely.                           */
/*--------------------------------------------------------------------*/
proc sort data=pinest1; by Symbol Month;
proc transpose data=pinest1 out=pinest2;
by Symbol Month;
	id Parameter;
	var Estimate;
quit;
data pinest2; set pinest2; drop _NAME_; run;
 
%if %sysfunc(exist(pinest2)) %then %do;
proc append data=pinest2 base=output.&COUNTRY._APINEST_MONTH; quit;
%end;

proc sql; drop table estdat, pinest1, pinest2; quit;
%end;
option notes;
%put Done!;
%mend APIN;

%APIN();

proc sql;
create table APIN as
	select distinct a.*, Month, a, d, t1, t2, ub, us, eb, es, pb, ps
	from input.&COUNTRY._Trade as a, output.&COUNTRY._APINEST_MONTH as b
	where a.Symbol=b.Symbol and intck("month", a.Date, b.Month)=-1;
quit;
data output.&COUNTRY._APIN_MONTH; set APIN;
APIN=( a*(d*ub+(1-d)*us) ) / ( a*(d*ub+(1-d)*us) + (pb+ps)*(a*t2+(1-a)*t1) + es + eb  );
PSOS=( (pb+ps)*(a*t2+(1-a)*t1) ) / ( a*(d*ub+(1-d)*us) + (pb+ps)*(a*t2+(1-a)*t1) + es + eb );
format APIN PSOS comma8.3;

Month=intnx("Month", Date, 0, 'e');
format Month yymmddn8.;
run;
/*
proc sort data=&OUTFILENAME; by Month;
proc means data=&OUTFILENAME noprint;
by Month;
	var APIN PSOS;
	output out=APINOUT mean=APIN PSOS;
quit;
*/
/* ********************************************************************************* */
/* *************  Material Copyright Wharton Research Data Services  *************** */
/* ****************************** All Rights Reserved ****************************** */
/* ********************************************************************************* */

