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
/* 1. Bring Monthly CRSP, TCLINK, and event date file */
/* 2. Merge them and leave permno, symbol, date, Volume Bucket Size */
/*     - Volume Bucket Size is determined by the number of trading volume */
/*        of each stocks in a month. I intend to get fifty observations of */
/*        VPIN metric per one day on average */
/***************************************************/

libname crsp "C:\Academic Area\Data from WRDS\CRSP\a_stock";
libname taq "C:\Academic Area\Data from WRDS\TAQ";
libname event "C:\Users\HaJinGi\Dropbox\SMU\Research\Low Frequency Order Imbalance\Data";
libname data "C:\Users\HaJinGi\Dropbox\SMU\Research\VPIN is informative\Data";

/* 1. Bring Monthly CRSP, TCLINK, and event date file */
*** Bring Monthly CRSP;
%let msevars=exchcd shrcd;
%let msfvars=prc vol cfacpr;

%include "C:\Academic Area\crspmerge.sas";
%include "C:\Academic Area\nwords.sas";

%crspmerge(s=m,start=01JAN1993, end=31DEC2013, sfvars=&msfvars,sevars=&msevars, filters=exchcd in (1,2,3) and shrcd in (10,11), outset = _crspm);
* the paper focuses on ordinary common shares of firms incoporated in the US that traded on the NYSE and AMEX;
data crspm; set _crspm; date=intnx("month", date, 0, 'e'); run;

*** Bring TCLINK orginal;
data TCLINK; set taq.tclink_199301_201312 (keep=permno symbol date); date=intnx("month", date, 0, 'e'); run;

/* 2. Merge them and leave permno, symbol, date, Volume Bucket Size */
/*     - Volume Bucket Size is determined by the number of trading volume */
/*        of each stocks in a month. I intend to get fifty observations of */
/*        VPIN metric per one day on average */
proc sort data=crspm; by permno date;
proc sort data=TCLINK; by permno date;
data VPIN_TCLINK; merge TCLINK (in=b) CRSPM (in=c);
by permno date; if b and c;

P=abs(prc)/cfacpr; if P>5; * Eliminate penny stocks; 
_vol=vol*100 /* Scale *//30 /* Average days a month *//50 /* Average Volume Bucket number a month */; 
V=(floor(_vol/10)+1)*10; /* Odd out */

keep permno symbol date V; run;

*** Save it;
proc sort data=VPIN_TCLINK out=data.TCLINK_whole nodup; by date symbol; run;

