/****************************************
Household effect size graphs - Figures 3 & S3
Dofile #3 in sequence
This do-file creates the effect-size grapsh for country analysis
Figures 3 and S3 in Science publication. Family outcomes
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

* Family indexed outcomes, country-level
use "${dta_working}\country_family_matrix.dta", clear

* Bringing in variable descriptions for graph (display text)
encode country, gen(country_num)
encode varname, gen(var_num)
loc suffix "eth gha hnd ind pak per"
sort country_num var_num 
loc varname "asset_index ind_fin ind_increv index_ctotal index_foodsecurity index_health index_mental index_political index_time index_women"
gen varname_ctry = ""
forvalues i=1/6 {
	forvalues j=1/10 {
		loc vname: word `j' of `varname'
		loc sfx: word `i' of `suffix'
		replace varname_ctry = "`vname'_`sfx'" if country_num==`i' & var_num==`j'
		}
	}
drop var_num
tempfile full values
save `full'
clear all
import excel "${dta}\\values_effectsize_country.xlsx", first clear
save `values'
merge 1:m varname using `full', nogen
sort country varname
drop varname

* Creating varnames equal to the family of outcome in which each variable belongs
loc ethiopia "index_ctotal_eth index_foodsecurity_eth asset_index_eth ind_fin_eth index_time_eth ind_increv_eth index_health_eth index_mental_eth index_political_eth index_women_eth"
loc ghana "index_ctotal_gha index_foodsecurity_gha asset_index_gha ind_fin_gha index_time_gha ind_increv_gha index_health_gha index_mental_gha index_political_gha index_women_gha"
loc honduras "index_ctotal_hnd index_foodsecurity_hnd asset_index_hnd ind_fin_hnd index_time_hnd ind_increv_hnd index_health_hnd index_mental_hnd index_political_hnd index_women_hnd"
loc india "index_ctotal_ind index_foodsecurity_ind asset_index_ind ind_fin_ind index_time_ind ind_increv_ind index_health_ind index_mental_ind index_political_ind index_women_ind"
loc pakistan "index_ctotal_pak index_foodsecurity_pak asset_index_pak ind_fin_pak index_time_pak ind_increv_pak index_health_pak index_mental_pak index_political_pak index_women_pak"
loc peru "index_ctotal_per index_foodsecurity_per asset_index_per ind_fin_per index_time_per ind_increv_per index_health_per index_mental_per index_political_per index_women_per"

* Creating variables for dimensions of the plot
gen area = ""
loc area_order 1
loc var_order 1

gen area_num = .
gen order = .
gen outcome_order = .

* Looping through Endline 1 and Endline 2, creating the graphs
loc ts "end fup"
local times ""Endline 1" "Endline 2""

forvalues k=1/2 {
		/* clearing old locals */
		foreach clear in lxs rxs textadd yline posclr negclr nsclr color msymbol estopts ciopts ///
		compltext comprtext inf opts  labelstext_lu labelstext_ld labelstext_ln labelstext_ru ///
		labelstext_rd labelstext_rn rangel ranger midpoint outliervar ylabel xlabel t time n i {
		loc `clear'
		}
	
	loc t :			word `k' of `ts' 
	loc time :		word `k' of `times'
	loc area_order 1
	loc var_order 1
	
	loc section_names ""Ethiopia" "Ghana" "Honduras" "India" "Pakistan" "Peru""
	loc sections ethiopia ghana honduras india pakistan peru
	if `k' == 2 {
		loc area_order 1
		loc var_order 1
		replace area = ""
		replace area_num = .
		replace order = .
		replace outcome_order = .
	}
	
	loc areacount: word count `sections'
	forvalues i = 1/`areacount' {
		loc outcome_order 1
		loc area_num `i'
		loc name: word `i' of `section_names'
		loc section: word `i' of `sections'
		foreach test in ``section'' {
			replace area = "`name'" if varname_ctry=="`test'"
			replace order = `var_order' if varname_ctry=="`test'"
			replace outcome_order = `outcome_order' if varname_ctry=="`test'"
			replace area_num = `area_num' if varname_ctry=="`test'"
			loc ++var_order
			loc ++outcome_order
			}
		}
	
	sort order
	if `k'==2 {
		drop if missing(outcome_order) & missing(area_num) // deleting empty rows at bottom!
	}
	isid area_num outcome_order
	
	* Creating horizontal lines between sections
	# de ;
	if `k'==1 { ;
	lab de area_num_lab
		1 "Ethiopia"
		2 "Ghana"
		3 "Honduras"
		4 "India"
		5 "Pakistan"
		6 "Peru" ;
	} ;
	# de cr
	lab values area_num area_num_lab // same label for both time periods

	loc n 0
	loc i 0

	while `++i' <= _N {
		if area[`i'] != area[`i' - 1] {
			loc ++n
			set obs `=_N + 2'
			replace order = order + 2 if _n >= `i'
			replace order = `i'		in `=_N - 1'
			replace order = `i' + 1	in L

			loc line_move_up 1.05
			loc ln`n' = cond(`i' == 1, 2.2, order[`i'] - 0.7) - `line_move_up'

			sort order
			loc i = `i' + 2
			display "`i'"
		}
	}

	assert `n'==`areacount'

	loc areacount: word count `sections'
	* Generating upper and lower confidence intervals for each outcome variable
	gen norm_CL_Treatment_`t' = itt_`t' - invttail(df_`t', .05) * se_`t'
	gen norm_CU_Treatment_`t' = itt_`t' + invttail(df_`t', .05) * se_`t'
	/* -------------------------------------------------------------------------- */
						/* removing missing variables for each round*/
						
	*Pakistan missing mental health in endline, India missing women's index in follow up
	preserve
	drop if missing(p_`t') & !missing(valuename)
	/* --------------------------------------------------------------------------- */

	* Interpretation of significance (to be used for color-coding graph)
	gen inference_`t' = ""
	replace inference_`t' = "Positive impact" if itt_`t' > 0 & ///
		p_`t' < .1
	replace inference_`t' = "Negative impact" if itt_`t' < 0 & ///
		p_`t' < .1
	replace inference_`t' = "Not significant" if p_`t' >= .1
	encode inference_`t', gen(inference_num_`t')

	* Creating scale for x-axis, width based on upper and lower bounds of min and max
	su norm_CL_Treatment_`t'
	loc lxs = min(r(min), -0.6)
	sum norm_CU_Treatment_`t'
	loc rxs = max(r(max), .5)

	* Area titles (-text()-)
	loc textadd text(
	forv i = 1/`areacount' {
		loc text_move_down 1.1
		loc p = `ln`i'' + `line_move_up' + `text_move_down'
		loc textadd `textadd' `p' `lxs' "`:lab area_num_lab `i''"
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
	/* -------------------------------------------------------------------------- */
						/* export graph			*/
	if `k'==1 {
		loc x S3
	}
	else if `k'==2 {
		loc x=3
	}
	loc gtitle Figure `x': Average Intent-to-Treat Effects by Country, `time' at a Glance

	#d ;
	loc caption "
		"This figure summarizes the treatment effects presented in Appendix Tables S3a-S3f.
		Here, all treatment effects are presented as standardized z-score indices."
		"Each line shows the standardized index outcome and its 95% confidence interval."
	";
	#d cr
	foreach lcl in gtitle caption {
		mata: st_local("`lcl'", stritrim(st_local("`lcl'")))
	}

	graph drop _all
	#d ;
	eclplot itt_`t' norm_CL_Treatment_`t' norm_CU_Treatment_`t' order,
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
		xlab(#15, labcolor(gs3) format(%9.1f))
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
	loc sgtitle Figure `x' `time'
	graph export "${figures}\`sgtitle'.ps", ///
		logo(off) pagesize(letter) mag(185) orientation(landscape) ///
		tmargin(.5) lmargin(.4) replace
	graph export "${figures}\`sgtitle'.wmf", ///
		fontface("Times New Roman") replace
	graph export "${figures}\`sgtitle'.pdf", ///
		replace
	restore
}	
