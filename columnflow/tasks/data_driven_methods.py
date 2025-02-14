
"""
Task to produce and merge histograms.
"""

from __future__ import annotations

import luigi
import law

from columnflow.tasks.framework.base import Requirements, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, VariablesMixin,
    ShiftSourcesMixin, WeightProducerMixin, ChunkedIOMixin, DatasetsProcessesMixin, CategoriesMixin
)
from columnflow.tasks.framework.plotting import ProcessPlotSettingMixin

from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.parameters import last_edge_inclusive_inst
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox, DotDict


class PrepareFakeFactorHistograms(
    WeightProducerMixin,
    MLModelsMixin,
    ProducersMixin,
    ReducedEventsUser,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    last_edge_inclusive = last_edge_inclusive_inst

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
    )

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    # names of columns that contain category ids
    # (might become a parameter at some point)
    category_id_columns = {"category_ids"}

    # register sandbox and shifts found in the chosen weight producer to this task
    register_weight_producer_sandbox = True
    register_weight_producer_shifts = True

    @law.util.classproperty
    def mandatory_columns(cls) -> set[str]:
        return set(cls.category_id_columns) | {"process_id"}

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.ProvideReducedEvents.req(self)

        if not self.pilot:
            if self.producer_insts:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]

            # add weight_producer dependent requirements
            reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))

        return reqs

    def requires(self):
        reqs = {"events": self.reqs.ProvideReducedEvents.req(self)}

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]

        # add weight_producer dependent requirements
        reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))

        return reqs

    workflow_condition = ReducedEventsUser.workflow_condition.copy()

    @workflow_condition.output
    def output(self):
        return {"hists": self.target(f"fake_factor__{self.branch}.pickle")}

    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import (
            Route, update_ak_array, add_ak_aliases, has_ak_column, fill_hist, EMPTY_FLOAT
        )

        # prepare inputs
        inputs = self.input()

        # declare output: dict of histograms
        histograms = {}

        # run the weight_producer setup
        producer_reqs = self.weight_producer_inst.run_requires()
        reader_targets = self.weight_producer_inst.run_setup(producer_reqs, luigi.task.getpaths(producer_reqs))

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})

        # define columns that need to be read
        read_columns = {Route("process_id")}
        read_columns |= set(map(Route, self.category_id_columns))
        read_columns |= set(self.weight_producer_inst.used_columns)
        read_columns |= set(map(Route, aliases.values()))
        read_columns |= {
            Route(the_ax.var_route) for the_ax in self.config_inst.x.fake_factor_method.axes.values()
        }
        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        # iterate over chunks of events and diffs
        file_targets = [inputs["events"]["events"]]
        if self.producer_insts:
            file_targets.extend([inp["columns"] for inp in inputs["producers"]])

        # prepare inputs for localization
        with law.localize_file_targets(
            [*file_targets, *reader_targets.values()],
            mode="r",
        ) as inps:
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.path for inp in inps],
                source_type=len(file_targets) * ["awkward_parquet"] + [None] * len(reader_targets),
                read_columns=(len(file_targets) + len(reader_targets)) * [read_columns],
                chunk_size=self.weight_producer_inst.get_min_chunk_size(),
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))

                # add additional columns
                events = update_ak_array(events, *columns)

                # add aliases
                events = add_ak_aliases(
                    events,
                    aliases,
                    remove_src=True,
                    missing_strategy=self.missing_column_alias_strategy,
                )

                # build the full event weight
                if hasattr(self.weight_producer_inst, "skip_func") and not self.weight_producer_inst.skip_func():
                    events, weight = self.weight_producer_inst(events)
                else:
                    weight = ak.Array(np.ones(len(events), dtype=np.float32))
                # define and fill histograms, taking into account multiple axes
                categories = self.config_inst.categories.ids()
                h = (hist.Hist.new
                    .IntCat(categories , name="category", growth=True)
                    .IntCat([], name="process", growth=True))
                for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                    h = eval(f'h.{var_axis.ax_str}') 
                
                histograms['fake_factors'] = h.Weight()
                
                category_ids = ak.concatenate(
                        [Route(c).apply(events) for c in self.category_id_columns],
                        axis=-1,
                    )
                # broadcast arrays so that each event can be filled for all its categories
                
                fill_data = {
                    "category"          : category_ids,
                    "process"           : events.process_id,
                    "weight"            : weight,
                }
                for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                    route = Route(var_axis.var_route)
                    if len(events) == 0 and not has_ak_column(events, route):
                        values = empty_f32
                    else:
                        values = ak.fill_none(ak.firsts(route.apply(events),axis=1), EMPTY_FLOAT)
                        if 'IntCategory' in var_axis.ax_str: values = ak.values_astype(values, np.int64)
                    fill_data[var_name] = values
                # fill it
                fill_hist(
                    histograms['fake_factors'],
                    fill_data,
                )
        # merge output files
        self.output()["hists"].dump(histograms, formatter="pickle")


# overwrite class defaults
check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
PrepareFakeFactorHistograms.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=PrepareFakeFactorHistograms.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


PrepareFakeFactorHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=PrepareFakeFactorHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)

class dict_creator():
    def init_dict(self, ax_list):
        if not ax_list:
            return -1.
        else:
            ax = ax_list[0]
            updated_ax = ax_list[1:]
            get_ax_dict = lambda ax, ax_list, func : {ax.bin(i): func(ax_list) for i in range(ax.size)}
            return get_ax_dict(ax,updated_ax, self.init_dict)
                

class ComputeFakeFactors(
    DatasetsProcessesMixin,
    CategoriesMixin,
    WeightProducerMixin,
    ProducersMixin,
    dict_creator,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )
    
    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        PrepareFakeFactorHistograms=PrepareFakeFactorHistograms,
    )
    
    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts
    
    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        if not self.pilot:
            variables = self._get_variables()
            if variables:
                reqs["ff_method"] = self.reqs.PrepareFakeFactorHistograms.req_different_branching(
                    self,
                    branch=-1,
                    variables=tuple(variables),
                )

        return reqs

    def requires(self):
        return {
            d: self.reqs.PrepareFakeFactorHistograms.req(
                self,
                dataset=d,
                branch=-1
            )
            for d in self.datasets
        }
    def output(self):
        return {"ff_json": self.target(f"fake_factors.json"),
                "plots": {'_'.join((ff_type, syst)): self.target(f"fake_factor_{ff_type}_{syst}.png")
                          for syst in ['nominal', 'up', 'down']
                          for ff_type in ['qcd','wj']},
                "plots1d": {'_'.join((ff_type,str(dm))): self.target(f"fake_factor_{ff_type}_PNet_dm_{str(dm)}.png")
                          for ff_type in ['qcd','wj']
                          for dm in [0,1,2,10,11]}}

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        from scipy.optimize import curve_fit
        from scipy.special import erf
        import matplotlib.pyplot as plt
        import correctionlib.schemav2 as cs
        plt.figure(dpi=200)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "monospace",
            "font.monospace": 'Computer Modern Typewriter'
        })
        # preare inputs and outputs
        inputs = self.input()
        outputs = self.output()
        merged_per_dataset = {}
        projected_hists = []
        hists_by_dataset = []
        for (dataset_name, dataset) in inputs.items():
            files = dataset['collection']
            # load input histograms per dataset
            hists_per_ds = [
                inp['hists'].load(formatter="pickle")['fake_factors']
                for inp in self.iter_progress(files.targets.values(), len(files), reach=(0, 50))
            ]
            ds_single_hist = sum(hists_per_ds[1:], hists_per_ds[0].copy())
            hists_by_dataset.append(ds_single_hist)
        #Create a dict of histograms indexed by the process
        hists_by_proc = {}
        for proc_name in self.config_inst.processes.names():
            proc = self.config_inst.processes.get(proc_name)
            for the_hist in hists_by_dataset:
                
                if proc.id in the_hist.axes["process"]: 
                    h = the_hist.copy()
                    h = h[{"process": hist.loc(proc.id)}]
                    # add the histogram
                    if proc in hists_by_proc:
                        hists_by_proc[proc] += h
                    else:
                        hists_by_proc[proc] = h
        
        #Divide histograms to data and bkg
        mc_hists    = [h for p, h in hists_by_proc.items() if p.is_mc and not p.has_tag("signal")]
        data_hists  = [h for p, h in hists_by_proc.items() if p.is_data]
        
        #Merge histograms to get a joint data and mc histogram
        if len(mc_hists) > 1:   mc_hists    = sum(mc_hists[1:], mc_hists[0].copy())
        else: mc_hists = mc_hists[0].copy()
        if len(data_hists) > 1: data_hists  = sum(data_hists[1:], data_hists[0].copy())
        else: data_hists = data_hists[0].copy()
        
        #Function that performs the calculation of th
        def get_ff_corr(self, h_data, h_mc, num_reg = 'dr_num_wj', den_reg = 'dr_den_wj', name='ff_hist', label='ff_hist'):
            def get_dr_hist(self, h, det_reg): 
                cat_name = self.categories[0]
                cat = self.config_inst.get_category(cat_name.replace('sr',det_reg))
                return h[{"category": hist.loc(cat.id)}]
            
            get_id = lambda ax, key: [i in enumerate(ax.keys)]
         
            data_num = get_dr_hist(self, h_data, num_reg)
            data_den = get_dr_hist(self, h_data, den_reg)
            mc_num = get_dr_hist(self, h_mc, num_reg)
            mc_den = get_dr_hist(self, h_mc, den_reg)
            
            num = data_num.values() - mc_num.values()
            den = data_den.values() - mc_den.values()
            ff_val = np.where((num > 0) & (den > 0),
                               num / np.maximum(den, 1),
                               -1)
            def rel_err(x):
                return x.variances()/np.maximum(x.values()**2, 1)
            
            ff_err = ff_val * ((data_num.variances() + mc_num.variances())**0.5 / np.abs(num) + (data_den.variances() + mc_den.variances())**0.5 / np.abs(den))
            
            
            h = hist.Hist.new
            for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                h = eval(f'h.{var_axis.ax_str}') 
            h = h.StrCategory(['nominal', 'up', 'down'], name='syst', label='Statistical uncertainty of the fake factor')
            ff_raw = h.Weight()
            ff_raw.view().value[...,0] = ff_val
            ff_raw.view().variance[...,0] = ff_err**2
            ff_raw.name = name + '_raw'
            ff_raw.label = label + '_raw'
            
            #Make an approximation of tau pt dependance
            formula_str = 'p0 + p1*x+p2*x*x'
            def fitf(x, p0, p1, p2):
                return eval(formula_str)
            def jac(x):
                from numpy import array
                out = array([[ 1, x,  x**2],[x,  x**2, x**3],[x**2,  x**3, x**4]])
                return out
            
            def eval_formula(formula_str, popt):
                for i,p in enumerate(popt):
                    formula_str = formula_str.replace(f'p{i}',str(popt[i]))
                return formula_str
            
            ff_fitted = ff_raw.copy().reset()
            ff_fitted.name = name
            ff_fitted.label = label
            fitres = {}
            
            axes = list(ff_raw.axes[1:2])
            fitres = {}
            dc = dict_creator()
            for the_field in ['chi2','ndf','popt', 'pcov', 'fitf_str']: 
                fitres[the_field]= dc.init_dict(axes)
            
            dm_axis = ff_raw.axes['tau_dm_pnet']
            for dm in dm_axis:
                h1d = ff_raw[{'tau_dm_pnet': hist.loc(dm),
                                'syst': hist.loc('nominal')}]
                mask = h1d.values() > 0
                y = h1d.values()[mask]
                y_err = (h1d.variances()[mask])**0.5
                x = h1d.axes[0].centers[mask]
                popt, pcov = curve_fit(fitf,x,y,
                                       sigma=y_err,
                                       absolute_sigma=True,
                                       )
                fitres['chi2'][dm] = sum(((y - fitf(x, *popt))/y_err)**2)
                fitres['ndf'][dm] = len(y) - len(popt)
                fitres['popt'][dm] = popt 
                fitres['pcov'][dm] = pcov
               
                fitres['fitf_str'][dm] = eval_formula(formula_str,popt)
                for c, shift_name in enumerate(['down', 'nominal', 'up']): # if down then c=-1, if up c=+1, nominal => c=0
                    ff_fitted.view().value[:,
                                           ff_fitted.axes[1].index(dm),
                                           ff_fitted.axes[2].index(shift_name)] = fitf(x, *popt + (c-1) * np.sqrt(np.diag(pcov)))
            fitres['name']  = name
            fitres['jac']   = jac
            fitres['fitf']  = fitf
            return ff_raw, ff_fitted, fitres
        
        wj_raw, wj_fitted, wj_fitres = get_ff_corr(self,
                              data_hists,
                              mc_hists,
                              num_reg = 'dr_num_wj',
                              den_reg = 'dr_den_wj',
                              name='ff_wjets',
                              label='Fake factor W+jets')
        
        qcd_raw, qcd_fitted, qcd_fitres = get_ff_corr(self,
                              data_hists,
                              mc_hists,
                              num_reg = 'dr_num_qcd',
                              den_reg = 'dr_den_qcd',
                              name='ff_qcd',
                              label='Fake factor QCD')
        
        corr_list = []
        for fitres in [wj_fitres, qcd_fitres]:
            formula_str = fitres['fitf_str']
            dm_bins = []
            for (dm, the_formula) in formula_str.items():
                x_max = 100
                last_val = fitres['fitf'](x_max,* fitres['popt'][dm])
                
                dm_bins.append(cs.CategoryItem(
                    key=dm,
                    value=cs.Formula(
                        nodetype="formula",
                        variables=["tau_pt"],
                        parser="TFormula",
                        expression=f'({the_formula})/(1. + exp(10.*(x-{x_max}))) + ({last_val})/(1. + exp(-10.*(x-{x_max})))',
                    )))
            corr_list.append(cs.Correction(
                name=fitres['name'],
                description=f"fake factor correcton for {fitres['name'].split('_')[1]}",
                version=2,
                inputs=[
                    cs.Variable(name="tau_pt", type="real",description="pt of tau"),
                    cs.Variable(name="tau_dm_pnet", type="int", description="PNet decay mode of tau"),
                ],
                output=cs.Variable(name="weight", type="real", description="Multiplicative event weight"),
                data=cs.Category(
                    nodetype="category",
                    input="tau_dm_pnet",
                    content=dm_bins,)
            ))
            
        cset = cs.CorrectionSet(
        schema_version=2,
        description="Fake factors",
        corrections=corr_list
        )
        self.output()['ff_json'].dump(cset.json(exclude_unset=True), formatter="json")
        
        
        
        #Plot fake factors:
        for h_name in ['wj', 'qcd']:
            h_raw = eval(f'{h_name}_raw')
            h_fitted = eval(f'{h_name}_fitted')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            h_raw[...,'nominal'].plot2d(ax=ax)
            self.output()['plots']['_'.join((h_name,'nominal'))].dump(fig, formatter="mpl")
            fitres = wj_fitres if h_name == 'wj' else qcd_fitres
            dm_axis = h_raw.axes['tau_dm_pnet']
            for dm in dm_axis:
                h1d = h_raw[{'tau_dm_pnet': hist.loc(dm),
                                'syst': hist.loc('nominal')}]
                hfit = h_fitted[{'tau_dm_pnet': hist.loc(dm)}]
                fig, ax = plt.subplots(figsize=(8, 6))
                mask = h1d.counts() > 0
                x = h1d.axes[0].centers[mask]
                y = h1d.counts()[mask]
                xerr = (np.diff(h1d.axes[0]).flatten()/2.)[mask],
                yerr = np.sqrt(h1d.variances()).flatten()[mask],
                ax.errorbar(x, y, xerr = xerr, yerr = yerr,
                                label=f"PNet decay mode = {dm}",
                                marker='o',
                                fmt='o',
                                line=None, color='#2478B7', capsize=4)
                x_fine = np.linspace(x[0],x[-1],num=100)
                popt = fitres['popt'][dm]
                pcov = fitres['pcov'][dm]
                jac = fitres['jac']
                def err(x,jac,pcov):
                    from numpy import sqrt,einsum
                    return sqrt(einsum('ij,ij',jac(x),pcov))

                import functools
                err_y = list(map(functools.partial(err, jac=jac,pcov=pcov), x_fine))
                
                y_fitf = fitres['fitf'](x_fine,*popt)
                y_fitf_up = fitres['fitf'](x_fine,*popt) + err_y
                y_fitf_down = fitres['fitf'](x_fine,*(popt)) - err_y
               
                ax.plot(x_fine,
                        y_fitf,
                        color='#FF867B')
                ax.fill_between(x_fine, y_fitf_up,  y_fitf_down, color='#83d55f', alpha=0.5)
                ax.set_ylabel('Fake Factor')
                ax.set_xlabel('Tau pT [GeV]')
                ax.set_title(f'Jet Fake Factors :Tau PNet Decay Mode {(dm)}')
                ax.annotate(rf"$\frac{{\chi^2}}{{ndf}} = \frac{{{np.round(fitres['chi2'][dm],2)}}}{{{fitres['ndf'][dm]}}}$",
                            (0.8, 0.9),
                            xycoords='axes fraction',
                            fontsize=20)
                
                self.output()['plots1d']['_'.join((h_name,str(dm)))].dump(fig, formatter="mpl")



class CreateDataDrivenHistograms(
    VariablesMixin,
    WeightProducerMixin,
    ProducersMixin,
    ReducedEventsUser,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ComputeFakeFactors=ComputeFakeFactors,
        ProduceColumns=ProduceColumns,
    )
    
    def requires(self):
        reqs = {"events": self.reqs.ProvideReducedEvents.req(self)}
        from IPython import embed; embed()
        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]
        reqs['ff_json'] = self.reqs.ComputeFakeFactors.req(self)
        reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))
        return reqs

    def output(self):
        return {"hists": self.target(f"histograms__vars_{self.variables_repr}__{self.branch}.pickle")}

    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import (
            Route, update_ak_array, add_ak_aliases, has_ak_column, fill_hist,
        )

        # prepare inputs
        inputs = self.input()
        from IPython import embed; embed()
        # declare output: dict of histograms
        histograms = {}