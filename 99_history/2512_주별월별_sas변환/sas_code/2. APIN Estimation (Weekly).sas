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
libname data "C:\TempSD";
libname temp "C:\TempSD";

%let Country=US;
%let UPDATE=01MAY2021;
/****************************************/

%let INFILENAME=&Country._TRADE;
%let OUTFILENAME=&Country._APIN_RWEEK4;

/*--------------------------------------------------------------------*/
/*    Get the data ready for the NLMXIED procedure.                       */
/*--------------------------------------------------------------------*/
data buysell; set data.&INFILENAME; 
Week=intnx("Week", Date, 0, 'b')+1;
format Week yymmddn8.;
keep Symbol Date Week B S; run;
data Week; set buysell;
keep Symbol Week; run;

proc sort data=buysell; by Symbol Week Date; 
proc sort data=Week nodup; by Symbol Week; quit;
proc sql;
create table _estdat as
	select distinct a.Symbol, a.Week, c.Date as RegDate, c.B, c.S
	from Week as a, buysell as c
	where a.Symbol=c.Symbol and -3<=intck("WEEK", a.Week, c.Date)<=0
	group by a.Symbol, a.Week having count(c.Date)>10;
quit;

/*--------------------------------------------------------------------*/
/*    This is the procedure used to maximize the log likelihood       */
/*    function specified below.                                       */
/*                                                                    */
/*    The parameters emerging from this estimation, for SYMBOL XYZ    */
/*    are consistent with what's mentioned in the Easley, Hvidjkaer,  */
/*    O'Hara paper.                                                   */
/*--------------------------------------------------------------------*/

proc sort data=_estdat; by Symbol; 
	where Week>="&UPDATE"d; quit;
data firm; set _estdat; keep Symbol; 
proc sort data=firm nodup; by Symbol; run;
%let NOBS=&SYSNOBS; * Do not move this line. It should follow the database of 'firm'.;

%macro PIN;
option nonotes;
%do FPRD=1 %to &NOBS;
%put &FPRD / &NOBS Processed...;

data target; set firm; if _n_=&FPRD; run;
data estdat; merge _estdat (in=a) target (in=t); if a and t; by Symbol; run;

proc printto log=junk2; run;
proc nlmixed data=estdat fd=central technique=quanew update=bfgs;
by Symbol Week;
	ods output ParameterEstimates=pinest1;
	parms a=.1 .5 .9, d=.1 .5 .9, t1=.1 .5 .9, t2=.1 .5 .9, 
				ub=20 200 2000, us=20 200 2000, 
				eb=20 200 2000, es=20 200 2000,
				pb=20 200 2000, ps=20 200 2000; 
	bounds 0 <= a d t1 t2 <= 1, ub us eb es pb ps >= 0;

	temp = (1-a)*(1-t1)*pdf('poisson',B,eb)*pdf('poisson',S,es)                    6
				+ (1-a)*t1*pdf('poisson',B,eb+pb)*pdf('poisson',S,es+ps)           5
				+ a*(1-t2)*(1-d)*pdf('poisson',B,eb)*pdf('poisson',S,us+es)        4
				+ a*t2*(1-d)*pdf('poisson',B,eb+pb)*pdf('poisson',S,us+es+ps)      3
				+ a*(1-t2)*d*pdf('poisson',B,ub+eb)*pdf('poisson',S,es)            2
				+ a*t1*d*pdf('poisson',B,ub+eb+pb)*pdf('poisson',S,es+ps);         1

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
quit;
proc printto; run;

/*--------------------------------------------------------------------*/
/*    Clean the output and print it nicely.                           */
/*--------------------------------------------------------------------*/
proc sort data=pinest1; by Symbol Week;
proc transpose data=pinest1 out=pinest2;
by Symbol Week;
	id Parameter;
	var Estimate;
quit;
data pinest2; set pinest2; drop _NAME_; run;
 
%if %sysfunc(exist(pinest2)) %then %do;
proc append data=pinest2 base=temp.&Country._apinest; quit;
%end;

proc sql; drop table estdat, pinest1, pinest2; quit;
%end;
option notes;
%put Done!;
dm 'odsresults; clear';
%mend;

%PIN();

proc sort data=temp.&Country._APINEST nodup; by Symbol Week;
proc sort data=data.&COUNTRY._Trade nodup; by Symbol Date; 
proc sql;
create table PIN as
	select distinct a.*, Week, a, d, t1, t2, ub, us, eb, es, pb, ps
	from data.&Country._Trade as a, temp.&Country._APINEST as b
	where a.Symbol=b.Symbol and intck("WEEK", a.Date, b.Week)=0;
	quit;

proc sort data=PIN nodup; by Symbol Date; quit;
data &OUTFILENAME; set PIN;
by Symbol Date;
	APIN=( a*(d*ub+(1-d)*us) ) / ( a*(d*ub+(1-d)*us) + (pb+ps)*(a*t2+(1-a)*t1) + es + eb  );
	PSOS=( (pb+ps)*(a*t2+(1-a)*t1) ) / ( a*(d*ub+(1-d)*us) + (pb+ps)*(a*t2+(1-a)*t1) + es + eb );
	format APIN PSOS comma8.3;

	if first.Date; * Delete duplicates;
	run;

proc sort data=&OUTFILENAME; by Week;
proc means data=&OUTFILENAME noprint;
by Week;
	var APIN PSOS;
	output out=APINOUT mean=APIN PSOS;
	quit;

/* ********************************************************************************* */
/* *************  Material Copyright Wharton Research Data Services  *************** */
/* ****************************** All Rights Reserved ****************************** */
/* ********************************************************************************* */

