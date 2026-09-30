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
libname output "C:\TempSD";

%let COUNTRY=US;
%let LOCKDOWN=14MAR2020;
/****************************************/

%macro PIN(INFILE, OUTFILE, WEEK);
option nonotes;
/*--------------------------------------------------------------------*/
/*    Get the data ready for the NLMXIED procedure.                       */
/*--------------------------------------------------------------------*/
data buysell; set input.&INFILE; 
where intnx("WEEK", "&WEEK"d, -3, 'b')<=Date<=intnx("WEEK", "&WEEK"d, 0, 'e');
Week="&WEEK"d; format Week yymmddn8.;
keep Symbol Week Date B S; run;

proc sort data=buysell; by Symbol Week Date; quit;
proc means data=buysell noprint;
by Symbol Week;
	var Date;
	output out=NObs (drop=_FREQ_ _TYPE_) n=NObs;
	* Number of non-zero trading days;
quit;

proc sort data=buysell; by Symbol Week;
proc sort data=NObs; by Symbol Week; quit;
data estdat; merge buysell (in=a) NObs (in=n); 
by Symbol Week; 

if a and n; 

if NObs>10; drop NObs;
* The number of non-zero trading days should be more than thirty days in a three month window;
run;


/*--------------------------------------------------------------------*/
/*    This is the procedure used to maximize the log likelihood       */
/*    function specified below.                                       */
/*                                                                    */
/*    The parameters emerging from this estimation, for SYMBOL XYZ    */
/*    are consistent with what's mentioned in the Easley, Hvidjkaer,  */
/*    O'Hara paper.                                                   */
/*--------------------------------------------------------------------*/

proc printto log=junk; run;
proc nlmixed data=estdat fd=central technique=quanew update=bfgs;
by Symbol Week;
	ods output ParameterEstimates=pinest1;
	parms a=.1 .5 .9, d=.1 .5 .9, u=20 200 2000, eb=20 200 2000, es=20 200 2000;
	bounds 0 <= a d <= 1, u eb es >= 0;

	temp = (1-a)*pdf('poisson',B,eb)*pdf('poisson',S,es)
				+ a*d*pdf('poisson',B,eb)*pdf('poisson',S,u+es)
				+ a*(1-d)*pdf('poisson',B,u+eb)*pdf('poisson',S,es);

	if temp = 0 then temp = 1E-300;
	loglik = log(temp);

	model B~general(loglik);

	estimate 'alpha' a;
	estimate 'delta' d;
	estimate 'mu' u;
	estimate 'epsilonb' eb;
	estimate 'epsilons' es;
quit;;
proc printto; run;
dm 'odsresults; clear';

/*--------------------------------------------------------------------*/
/*    Clean the output and print it nicely.                           */
/*--------------------------------------------------------------------*/
proc transpose data=pinest1 out=pinest2;
by Symbol Week;
	id Parameter;
	var Estimate;
quit;
data pinest2; set pinest2; drop _NAME_; run;
 
%if %sysfunc(exist(pinest2)) %then %do;
proc append data=pinest2 base=output.&OUTFILE; quit;
%end;

proc sql; drop table buysell, estdat, pinest1, pinest2, nobs; quit;
option notes;
%put Done!;
%mend PIN;

%macro WeeklyPIN(fdate,ldate);
/* set string as date format */
%let fdated = %sysfunc(InputN(&fdate,yymmdd8.));
%let ldated = %sysfunc(InputN(&ldate,yymmdd8.));

/* start loop in target period */
%do date = &fdated %to &ldated;
	%let ndate= %sysfunc(PutN(&date,date9));
	%let wkday= %sysfunc(weekday(&date));

	/*if wct is not missing, then do the following calculation*/
	%if %index(2,&wkday)>0 %then 
	%do;
		%put Processing &ndate...;
		%PIN(&COUNTRY._TRADE, &COUNTRY._PINEST, &ndate);
	%end;
%end;
%mend;
%WeeklyPIN(20200623, 20210731);

proc sort data=output.&COUNTRY._PINEST nodup; by Symbol Week;
proc sort data=input.&COUNTRY._Trade nodup; by Symbol Date;
proc sql;
create table _PIN as
	select distinct a.*, b.Week, a, d, u, eb, es
	from input.&COUNTRY._Trade as a, output.&COUNTRY._PINEST as b
	where a.Symbol=b.Symbol and intck("WEEK", a.Date, b.Week)=0;
	quit;

proc sort data=_PIN nodup; by Symbol Date; quit;
data &COUNTRY._PIN_RWEEK4; set _PIN;
by Symbol Date;
	PIN=a*u/(a*u + eb + es);
	if first.Date; * Delete duplicates;
	run;

proc sort data=&COUNTRY._PIN_RWEEK4; by Week;
proc means data=&COUNTRY._PIN_RWEEK4 noprint;
by Week;
	var PIN;
	output out=PINOUT mean=PIN;
	quit;

/* ********************************************************************************* */
/* *************  Material Copyright Wharton Research Data Services  *************** */
/* ****************************** All Rights Reserved ****************************** */
/* ********************************************************************************* */

