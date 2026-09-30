/***************************************************/
/***************************************************/
/* ***** PhD Term Project - VPIN is informative?  ***** */
/***************************************************/
/***************************************************/
/* Purpose: This program will generate TCLINK file to connect */
/*                TAQ and CRSP via PERMNO for paritcular stock-day */
/*                when it experiences extreme price change */
/*                 It also provides volume bucket size for VPIN measure */
/* Requirement: TCLINK original file and Monthly CRSP */
/* Pseudo Code: */
/* = This procedure is repeated for every trading day = */
/* 1. Bring Monthly TAQ which is matched with stock-day */
/*    - Leave observations if there is PERMNO and SIZE */
/*    - The size of Volume Bucket is updated monthly */
/*    - and if trade time is between "09:00:00"t and "16:00:00"t and price>0 and size>0 */
/*       corr in (0,7,8,9) and cond not in ( "O" "Z" "T" "L" "W") */
/* 2. Make one-minute time bar, and get transaction price */
/*     of last trade in each time bar, total trading volume in each time bar */
/* 3. Bring initial-value data file and merge it with current TAQ file */
/*     - Excess volume is re-located into the next bucket */
/*     - If a bucket is filled, then save TimeLine, Price, and Remained Volume into a file, __VPIN */
/* 4. Compute daily standard deviation of price change */
/*    - Stdev of Prc Ch. is updated daily if the daily num. of obs is more than ten. */
/* 5. Update initial-value file */
/***************************************************/

libname WTAQ "C:\Academic Area\Data from WRDS\WTAQ";
libname data "C:\Users\HaJinGi\Dropbox\SMU\Research\VPIN is informative\Data";
libname temp "C:\Academic Area\Temp";

*libname WTAQ '/wrds/nyse/sasdata/wrds_taqs_ct'; *TAQ trade and NBBO matched data from WRDS Cloud;
*libname data '/home/smusg/jingiha'; *home;
*libname temp '/scratch/smusg'; *temp space;

*** Target Period;
%let BEG=19930101;
%let END=19930131;

/* 1. Bring Monthly TAQ which is matched with stock-day and make initial-value file */
/*    - Leave observations if there is PERMNO and SIZE */
/*    - The size of Volume Bucket is updated monthly */
/*    - and if trade time is between "09:00:00"t and "16:00:00"t and price>0 and size>0 */
/*       corr in (0,7,8,9) and cond not in ( "O" "Z" "T" "L" "W") */
*** Make initial-value data file;
data initial; 
input symbol $ Initial_P Initial_Bucket Initial_TAU;
datalines;
RPICA . 0 0
;
run;

*** Bring Monthly TAQ and match to TCLINK for VPIN;
proc sort data=data.tclink_whole out=tclink; by symbol date; run;

%macro VPIN_GENERATOR(fdate, ldate);
*** Set string as date format;
%let fdated = %sysfunc(InputN(&fdate,yymmdd8.));
%let ldated = %sysfunc(InputN(&ldate,yymmdd8.));

*** Loop start;
%do date = &fdated %to &ldated;

%let ndate= %sysfunc(PutN(&date,yymmddn8));
%let wkday= %sysfunc(weekday(&date));
%let dct = wtaq.wct_&ndate. ; /* TRADES */

%let dout = _VPIN; 

/*if wct is not missing, then do the following calculation*/
%if %index(2 3 4 5 6,&wkday)>0 %then 
	%if %sysfunc(exist(&dct)) %then
		%do;
data __TAQ; set &dct; datem=intnx("month", date, 0, 'e'); format datem date9.; run;
data _TAQ; merge tclink (in=a rename=(date=datem)) __TAQ (in=b);  by symbol datem; 
if not missing(size) and not missing(permno); 
if "09:00:00"t<time and time<"16:00:00"t and price>0 and size>0 
			and corr in (0,7,8,9) and cond not in ( "O" "Z" "T" "L" "W");
timem=intnx("minute", time, 0, 'b');
format timem time.;
keep permno date time timem symbol V PRICE SIZE;
run;

/* 2. Make one-minute time bar, and get transaction price */
/*     of last trade in each time bar, total trading volume in each time bar */
proc sort data=_TAQ; by symbol date timem time;
data TAQ; set _TAQ; by symbol date timem time;

retain Vi;

if first.timem then Vi=0; 
Vi+SIZE; 
if last.timem;

drop SIZE time;
run;

/* 3. Merge a initial-value file with a current TAQ file */
/*     - Excess volume is re-located into the next bucket */
/*     - If a bucket is filled, then save TimeLine, Price, and Remained Volume into a file, __VPIN */

*** Merge it with current TAQ file;
proc sort data=TAQ; by symbol date timem; run;

** Match initial value to target stock;
data _VolBucket; merge TAQ (in=a) initial; by symbol; if a; run;

** Set initlal value to target stock-day;
data VolBucket __VPIN; set _VolBucket; 
by symbol date;

retain P TAU Bucket;

* Intialization;
if first.symbol then 
	if missing(Initial_P+Initial_TAU+Initial_Bucket) then 
	do; 
		P=PRICE; TAU=0; Bucket=0; DP=0; 
	end;
	else
	do;
		P=Initial_P; TAU=Initial_TAU; Bucket=initial_Bucket; DP=0;
	end;

* Volume bucket;
Bucket+Vi;
DP=P-PRICE;

* Excess volume is re-located into the next bucket;
if Bucket>V then 
	do;
	* If a bucket is filled, then save TimeLine, Price, and Remained Volume into a file, __VPIN;
	do until (Bucket<V);
		Bucket=Bucket-V;
		TAU+1;
		output __VPIN;
	end;
	P=PRICE;
	end;
output VolBucket;
drop Initial_P initial_TAU initial_Bucket;
run;

/* 4. Compute daily standard deviation of price change */
/*    - Stdev of Prc Ch. is updated daily if the daily num. of obs is more than ten. */
proc means data=VolBucket noprint;
by symbol date;
	var DP;
	output out=STDEV std=stdev;
run;

*** Merge stdev file with VPIN file;
proc sort data=__VPIN; by symbol date;
proc sort data=STDEV; by symbol date; run;
data &dout; merge __VPIN STDEV; 
by symbol date; if _FREQ_>10;

if stdev ne 0 then do; Buys=V*CDF("NORMAL", DP/STDEV); OI=abs(2*Buys-V); end;
drop _TYPE_ _FREQ_ Bucket PRICE P SIZE;
run;

/* 5. Update initial-value file */
data _initial; set VolBucket; by symbol date; if last.symbol; keep symbol P Bucket TAU; run;

proc sort data=initial; by symbol;
proc sort data=_initial; by symbol;
data initial; merge initial (in=a) _initial (in=b); by symbol; if a or b;
if not missing(P) then Initial_P=P;
if not missing(TAU) then Initial_TAU=TAU;
if not missing(Bucket) then Initial_Bucket=Bucket;
drop P Bucket TAU;
run;
%end;
%if %sysfunc(exist(&dout))%then
	%do;
	proc append base=VPIN data=&dout; run;
	proc sql; drop table stdev, taq, volbucket, _initial, _taq, _volbucket, __vpin; quit;
	proc sql; drop table &dout; quit;
	%end;
%end;
%mend VPIN_GENERATOR;

%VPIN_GENERATOR(&BEG, &END);
proc sort data=VPIN nodup; by symbol date tau; run;

proc printto log=junk; run;
proc expand data=VPIN out=_VPIN_&BEG._&END (drop=Buys OI DP Stdev) method=none;
by symbol;
	id tau;
	convert OI=TOI / transformout=(MOVSUM 50 trimleft 49);
quit;
proc printto; run;
proc sql; drop table VPIN; quit;

data VPIN_&BEG._&END; set _VPIN_&BEG._&END;
VPIN=TOI/(50*V);

if not missing(VPIN);
drop TOI V;
run;
proc sort data=VPIN_&BEG._&END; by permno date timem tau; run;
data temp.MinVPIN_&BEG._&END._wh; set VPIN_&BEG._&END; by permno date timem tau; if last.timem; drop TAU SYMBOL; run;
proc means data=temp.MinVPIN_&BEG._&END._wh noprint;
by permno date;
	var VPIN;
	output out=temp.DailyVPIN_&BEG._&END._wh (drop=_TYPE_ _FREQ_) 
		mean=VPIN_M std=VPIN_S max=VPIN_MAX min=VPIN_MIN;
run;

