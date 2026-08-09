/****************************************
Household effect size graphs - Figures 2 & S2
Dofile #3 in sequence
This do-file creates the effect-size grapsh for pooled analysis
Figures 2 and S2 in Science publication
Stata version: 12.1
*****************************************/

clear all
set maxvar 15000
cap log close
version 12.1

if "${master_test}!"!="1" {
	cap do "1_set_globals.do"
	if _rc==601 {
		di "Please run the 1_set_globals do-file before continuing"
	}
}

* Merge effect sizes from household level and adult level data into one dataset
use "${dta_working}\outcome_matrix_hh.dta", clear
append using "${dta_working}\outcome_matrix_ad.dta"
save "${dta_working}\effect_sizes.dta", replace

* Bringing in variable descriptions for graph (display text)
encode varname, gen(varname_num)
tempfile full values
save `full'
clear all
import excel "${dta}\\value_names_effect_sizes.xlsx", first clear
save `values'
merge 1:1 varname using `full'
assert _merge==3
drop _merge

* Creating varnames equal to the family of outcome in which each variable belongs
loc consumption "ctotal_pcmonth cfood_pcmonth cnonfood_pcmonth cdurable_pcmonth"
loc food_security "index_foodsecurity fs_enoughfood fs_adultskip fs_wholeday fs_childskip fs_twomeals"
loc assets "asset_index asset_tot_value asset_prod_index asset_prod_value asset_hh_index asset_hh_value"
loc finance "loan_totalamt loan_informalamt loan_formalamt sav_totalamt sav_depositamt"
loc time_use "time_work time_agri time_animals time_business time_paidlabor"
loc increv "ranimals_month iagri_month ibusiness_month ipaidlabor_month percep_econ"
loc physical "index_health ddays_nomiss dailyliving_score percep_health"
loc mental "index_mentalhealth percep_life index_stress dnoworry"
loc political "index_political pol_vote pol_party pol_leadermeet pol_leaderask"
loc women "index_women dec_food_major dec_educ_major dec_health_major dec_himprov_major dec_hmgmt_major"

* Creating variables for dimensions of the plot
gen area = ""
loc section_names ""Consumption" "Food Security" "Assets" "Finance" "Time Use" "Income and Revenues" "Physical Health" "Mental Health" "Political Involvement" "Women's Decision-making""
loc sections consumption food_security assets finance time_use increv physical mental political women

loc area_order 1
loc var_order 1

* Setting dimensions and order of variables
gen area_num = .
gen order = .
gen outcome_order = .
loc areacount: word count `sections'
forvalues i = 1/`areacount' {
	loc outcome_order 1
	loc area_num `i'
	loc name: word `i' of `section_names'
	loc section: word `i' of `sections'
	foreach test in ``section'' {
		replace area = "`name'" if varname=="`test'"
		replace order = `var_order' if varname=="`test'"
		replace outcome_order = `outcome_order' if varname=="`test'"
		replace area_num = `area_num' if varname=="`test'"
		loc ++var_order
		loc ++outcome_order
	}
}

sort order

* Creating horizontal lines between sections
isid area_num outcome_order // must be unique ID, code will break if this is not the case
# de ;
lab de area_num 
1 "Consumption"
2 "Food Security"
3 "Assets"
4 "Finance"
5 "Time Use"
6 "Income and Revenues"
7 "Physical Health"
8 "Mental Health"
9 "Political Involvement"
10 "Women's Decision-making"
;
# de cr
lab values area_num area_num

loc n 0
loc i 0
while `++i' <= _N {
	if area[`i'] != area[`i' - 1] {
		loc ++n
		
		set obs `=_N + 2'
		replace order = order + 2 if _n >= `i'
		replace order = `i'		in `=_N - 1'
		replace order = `i' + 1	in L

		loc line_move_up 1.5
		loc ln`n' = cond(`i' == 1, 2.2, order[`i'] - 0.7) - `line_move_up'

		sort order
		loc i = `i' + 2
	}
}

assert `n'==`areacount'

* Looping through Endline 1 and Endline 2, creating the graphs
loc ts "end fup"
local times ""Endline 1" "Endline 2""

forvalues k=1/2 {
	loc t :			word `k' of `ts' 
	loc time :		word `k' of `times'

	foreach clear in lxs rxs textadd yline posclr negclr nsclr color msymbol estopts ciopts ///
		compltext comprtext inf opts  labelstext_lu labelstext_ld labelstext_ln labelstext_ru ///
		labelstext_rd labelstext_rn rangel ranger midpoint outliervar {
		loc `clear'
	}
	loc areacount: word count `sections'
	* Generating upper and lower confidence intervals for each outcome variable
	gen norm_CL_Treatment_`t' = B_Treatment_`t' - invttail(DF_`t', .05) * SE_Treatment_`t'
	gen norm_CU_Treatment_`t' = B_Treatment_`t' + invttail(DF_`t', .05) * SE_Treatment_`t'

	* Interpretation of significance (to be used for color-coding graph)
	gen inference_`t' = ""
	replace inference_`t' = "Positive impact" if B_Treatment_`t' > 0 & ///
		p_value_Treatment_`t' < .1
	replace inference_`t' = "Negative impact" if B_Treatment_`t' < 0 & ///
		p_value_Treatment_`t' < .1
	replace inference_`t' = "Not significant" if p_value_Treatment_`t' >= .1
	encode inference_`t', gen(inference_num_`t')

	* Creating scale for x-axis, width based on upper and lower bounds of min and max
	su norm_CL_Treatment_`t'
	loc lxs = min(r(min), -0.6)
	sum norm_CU_Treatment_`t'
	loc rxs = max(r(max), .4)

	* Area titles (-text()-)
	loc textadd text(
	forv i = 1/`areacount' {
		loc text_move_down 0.6
		loc p = `ln`i'' + `line_move_up' + `text_move_down'
		loc textadd `textadd' `p' `lxs' "`:lab area_num `i''"
	}
	loc textadd `textadd', margin(small) place(1) just(left) color(black))

	* Horizontal lines (-yline()-)
	loc yline yline(
	forv i = 1/`areacount' {
		loc yline `yline' `ln`i''
	}
	loc yline `yline', lcolor(gs4))

	* Color constants
	* "clr" suffix for "color": "posclr" for "positive (impact) color."
	loc posclr black
	loc negclr black
	* "ns" for "not significant"
	loc nsclr gs8
	
	* Assigning symbols to significance
	tab inference_num_`t'
	* -estopts#()- and -ciopts#()-
	forv i = 1/`r(r)' {
		loc lab`i' : lab inference_num_`t' `i'
		if "`lab`i''" == "Positive impact" {
			loc color `posclr'
			loc msymbol D // diamond
		}
		else if "`lab`i''" == "Negative impact" {
			loc color `negclr'
			loc msymbol D // diamond
		}
		else if "`lab`i''" == "Not significant" {
			loc color `nsclr'
			loc msymbol Dh // diamond hollow
		}
		else {
			di as err "invalid inference_num value label"
			ex 9
		}

		loc estopts `estopts' estopts`i'(msymbol(`msymbol') mcolor(`color'))
		loc ciopts  `ciopts'  ciopts`i'(lpattern(solid) lcolor(`color'))
	}
	* Added text
	loc compltext " "
	loc comprtext " "
	forv j = 1/`=_N' {
		if valuename[`j'] != "" {
			loc d = cond(norm_CL_Treatment_`t'[`j'] - ///
				length(valuename[`j']) / 140 > `lxs', "l", "r")
			#d ;
			loc inf =
				cond(inference_`t'[`j'] == "Positive impact",  "u",
				cond(inference_`t'[`j'] == "Negative impact",  "d",
				cond(inference_`t'[`j'] == "Not significant",  "n", "")))
			;
			#d cr
			assert "`inf'" != ""
			loc labelstext_`d'`inf' `labelstext_`d'`inf'' ///
				`=order[`j'] + .2' `=norm_CL_Treatment_`t'[`j']' ///
				"`=valuename[`j']'"
		}
	}
	* `comp?text'
	loc compltext `compltext', ///
		margin(right) place(9) just(right) align(bottom) size(vsmall)
	loc comprtext `comprtext', ///
		margin(left) place(3) just(left) align(bottom) size(vsmall)
	* `labelstext_l?'
	loc opts margin(right) place(9) just(right) align(bottom) size(vsmall)
	loc labelstext_lu `labelstext_lu', color(`posclr') `opts'
	loc labelstext_ld `labelstext_ld', color(`negclr') `opts'
	loc labelstext_ln `labelstext_ln', color(`nsclr') `opts'

	* `labelstext_r?'
	loc margin(left) opts place(3) just(left) align(bottom) size(vsmall)
	loc labelstext_ru `labelstext_ru', color(`posclr') `opts'
	loc labelstext_rd `labelstext_rd', color(`negclr') `opts'
	loc labelstext_rn `labelstext_rn', color(`nsclr') `opts'


	/* --------------------EXPORTING GRAPH------------------------------- */
	
	if `k'==1 {
		loc num S2
	}
	else if `k'==2 {
		loc num = 2
	}

	loc gtitle Figure `num': Pooled Average Intent-to-Treat Effects, `time' at a Glance

	#d ;
	loc caption "
		"This figure summarizes the treatment effects presented in Tables S2a- S2h.
		Here, treatment effects on continuous variables are presented in
		standard deviation units."
		"Each line shows the OLS point estimate and 95% confidence interval for
		that outcome."
	";
	#d cr
	foreach lcl in gtitle caption {
		mata: st_local("`lcl'", stritrim(st_local("`lcl'")))
	}

	graph drop _all
	#d ;
	eclplot B_Treatment_`t' norm_CL_Treatment_`t' norm_CU_Treatment_`t' order,
		supby(inference_num_`t', spaceby(0.1)) `estopts' `ciopts'
		horizontal plotregion(style(none) color(white))
		graphregion(style(none) color(white)) scale(.5)
		/* title, etc. */
		title("`gtitle'", margin(medlarge) align(top) size(medlarge) color(gs1*10))
		caption(`caption', size(small)) legend(off)
		/* y-axis */
		ytitle("") yscale(noline) ylab(none, noticks)
		/* x-axis */
		xtitle("Effect size in standard deviations of the control group",
			margin(small) color(gs3))
		xscale(range(`lxs'/`rxs') lcolor(gs3))
		xlab(#15, labcolor(gs3) format(%9.1f)) //MD changed back to #15 instead of `xlabel' 8/26/2014
		/* added lines */
		`yline' xline(0 `midpoint', lpattern(dash) lwidth(vthin) lcolor(gs8*1.2))
		/* added text */
		`textadd'
		text(`labelstext_lu') text(`labelstext_ld')
		text(`labelstext_ln') text(`labelstext_ru') 
		text(`labelstext_rd') text(`labelstext_rn')
		text(`compltext')     text(`comprtext')
	;
	#d cr
	loc sgtitle Figure `num' `time'
	graph export "${figures}\`sgtitle'.ps", ///
		logo(off) pagesize(letter) mag(185) orientation(landscape) ///
		tmargin(.5) lmargin(.4) replace
	graph export "${figures}\`sgtitle'.wmf", ///
		fontface("Times New Roman") replace
	graph export "${figures}\`sgtitle'.pdf", ///
		replace
}	
