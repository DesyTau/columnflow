
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
    VariablesMixin,
    WeightProducerMixin,
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

class ComputeFakeFactors(
    VariablesMixin,
    DatasetsProcessesMixin,
    CategoriesMixin,
    WeightProducerMixin,
    ProducersMixin,
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
                branch=-1,
            )
            for d in self.datasets
        }
    def output(self):
        return {"ff_json": {ff_type: self.target(f"fake_factors_{ff_type}.json")for ff_type in ['qcd','wj']},
                "plots": {'_'.join((ff_type, syst)): self.target(f"fake_factor_{ff_type}_{syst}.png")
                          for syst in ['nominal', 'up', 'down']
                          for ff_type in ['qcd','wj']},}

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        import matplotlib.pyplot as plt
        import correctionlib.convert as cl_convert 
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
            self.publish_message(f"merging Fake factor histograms for {dataset_name}")
            ds_single_hist = sum(hists_per_ds[1:], hists_per_ds[0].copy())
            hists_by_dataset.append(ds_single_hist)
        #Create a dict of histograms indexed by the process
        hists_by_proc = {}
        for proc_name in self.config_inst.processes.names():
            proc = self.config_inst.processes.get(proc_name)
            self.publish_message(f"merging Fake factor histograms for process: {proc.name}")
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
        if len(data_hists) > 1: data_hists  = sum(data_hists[1:], data_hists[0].copy())
        
        #Function that performs the calculation of th
        def get_ff_corr(self, h_data, h_mc, num_reg = 'dr_num_wj', den_reg = 'dr_den_wj', name='ff_hist', label='ff_hist'):
            def get_dr_hist(self, h, det_reg): 
                cat_name = self.categories[0]
                from IPython import embed; embed()
                cat = self.config_inst.get_category(cat_name.replace('sr',det_reg))
                return h[{"category": hist.loc(cat.id)}]
         
            data_num = get_dr_hist(self, h_data, num_reg)
            data_den = get_dr_hist(self, h_data, den_reg)
            mc_num = get_dr_hist(self, h_mc, num_reg)
            mc_den = get_dr_hist(self, h_mc, den_reg)
            
            num = data_num.values() - mc_num.values()
            den = data_den.values() - mc_den.values()
            ff_val = np.where((num > 0) & (den > 0),
                               num / np.maximum(den, 1),
                               1)
            def rel_err(x):
                return x.variances()/np.maximum(x.values()**2, 1)
            
            ff_err2 = np.where((num > 0) & (den > 0),
                               np.sqrt(rel_err(data_num) + 
                                       + rel_err(data_den) +
                                       + rel_err(mc_num) + 
                                       + rel_err(mc_den)) * ff_val**2,
                               0.5* np.ones_like(ff_val))
            h = hist.Hist.new
            for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                h = eval(f'h.{var_axis.ax_str}') 
            h = h.StrCategory(['nominal', 'up', 'down'], name='syst', label='Statistical uncertainty of the fake factor')
            ff_hist= h.Weight()
            ff_hist.view().value[...,0] = ff_val
            ff_hist.view().value[...,1] = ff_val + np.sqrt(ff_err2)
            ff_hist.view().value[...,2] = np.maximum(ff_val - np.sqrt(ff_err2),0)
            ff_hist.name = name
            ff_hist.label = label
            ff_corr = cl_convert.from_histogram(ff_hist)
            ff_corr.data.flow = "clamp"
            return ff_corr, ff_hist
        
        wj_corr, wj_h = get_ff_corr(self,
                              data_hists,
                              mc_hists,
                              num_reg = 'dr_num_wj',
                              den_reg = 'dr_den_wj',
                              name='ff_wjets',
                              label='Fake factor W+jets')
        
        qcd_corr, qcd_h = get_ff_corr(self,
                              data_hists,
                              mc_hists,
                              num_reg = 'dr_num_qcd',
                              den_reg = 'dr_den_qcd',
                              name='ff_qcd',
                              label='Fake factor QCD')
        
        for h_name in ['wj', 'qcd']:
            the_hist = eval(f'{h_name}_h')
            
            for syst in ['nominal','up','down']:
                fig, ax = plt.subplots(figsize=(12, 8))
                the_hist[...,syst].plot2d(ax=ax)
                self.output()['plots']['_'.join((h_name,syst))].dump(fig, formatter="mpl")
                
        self.output()['ff_json']['wj'].dump(wj_corr.json(exclude_unset=True), formatter="json")
        self.output()['ff_json']['qcd'].dump(qcd_corr.json(exclude_unset=True), formatter="json")



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

#         # run the weight_producer setup
#         producer_reqs = self.weight_producer_inst.run_requires()
#         reader_targets = self.weight_producer_inst.run_setup(producer_reqs, luigi.task.getpaths(producer_reqs))

#         # create a temp dir for saving intermediate files
#         tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
#         tmp_dir.touch()

#         # get shift dependent aliases
#         aliases = self.local_shift_inst.x("column_aliases", {})

#         # define columns that need to be read
#         read_columns = {Route("process_id")}
#         read_columns |= set(map(Route, self.category_id_columns))
#         read_columns |= set(self.weight_producer_inst.used_columns)
#         read_columns |= set(map(Route, aliases.values()))
#         read_columns |= {
#             Route(inp)
#             for variable_inst in (
#                 self.config_inst.get_variable(var_name)
#                 for var_name in law.util.flatten(self.variable_tuples.values())
#             )
#             for inp in ((
#                 {variable_inst.expression}
#                 if isinstance(variable_inst.expression, str)
#                 # for variable_inst with custom expressions, read columns declared via aux key
#                 else set(variable_inst.x("inputs", []))
#             ) | (
#                 # for variable_inst with selection, read columns declared via aux key
#                 set(variable_inst.x("inputs", []))
#                 if variable_inst.selection != "1"
#                 else set()
#             ))
#         }

#         # empty float array to use when input files have no entries
#         empty_f32 = ak.Array(np.array([], dtype=np.float32))

#         # iterate over chunks of events and diffs
#         file_targets = [inputs["events"]["events"]]
#         if self.producer_insts:
#             file_targets.extend([inp["columns"] for inp in inputs["producers"]])
#         # if self.ml_model_insts:
#         #     file_targets.extend([inp["mlcolumns"] for inp in inputs["ml"]])

#         # prepare inputs for localization
#         with law.localize_file_targets(
#             [*file_targets, *reader_targets.values()],
#             mode="r",
#         ) as inps:
#             for (events, *columns), pos in self.iter_chunked_io(
#                 [inp.abspath for inp in inps],
#                 source_type=len(file_targets) * ["awkward_parquet"] + [None] * len(reader_targets),
#                 read_columns=(len(file_targets) + len(reader_targets)) * [read_columns],
#                 chunk_size=self.weight_producer_inst.get_min_chunk_size(),
#             ):
#                 # optional check for overlapping inputs
#                 if self.check_overlapping_inputs:
#                     self.raise_if_overlapping([events] + list(columns))

#                 # add additional columns
#                 events = update_ak_array(events, *columns)

#                 # add aliases
#                 events = add_ak_aliases(
#                     events,
#                     aliases,
#                     remove_src=True,
#                     missing_strategy=self.missing_column_alias_strategy,
#                 )

#                 # build the full event weight without fake factors
#                 if hasattr(self.weight_producer_inst, "skip_func") and not self.weight_producer_inst.skip_func():
#                     events, weight = self.weight_producer_inst(events)
#                 else:
#                     weight = ak.Array(np.ones(len(events), dtype=np.float32))

#                 # define and fill histograms, taking into account multiple axes
#                 for var_key, var_names in self.variable_tuples.items():
#                     # get variable instances
#                     variable_insts = [self.config_inst.get_variable(var_name) for var_name in var_names]

                    
#                     # create the histogram if not present yet
#                     if var_key not in histograms:
#                         for reg_key in ['ar_wj','ar_wj','ar_yields']:
#                             h = (
#                                 hist.Hist.new
#                                 .IntCat([], name="process", growth=True)
#                                 .IntCat([], name="shift", growth=True)
#                             )
#                             # add variable axes
#                             for variable_inst in variable_insts:
#                                 h = h.Var(
#                                     variable_inst.bin_edges,
#                                     name='_'.join((variable_inst.name, reg_key))
#                                     label=variable_inst.get_full_x_title(),
#                                 )
#                             # enable weights and store it
#                             histograms[var_key] = h.Weight()
                            
#                     # merge category ids
#                     category_ids = ak.concatenate(
#                         [Route(c).apply(events) for c in self.category_id_columns],
#                         axis=-1,
#                     )

#                     # broadcast arrays so that each event can be filled for all its categories
#                     fill_data = {
#                         "category": category_ids,
#                         "process": events.process_id,
#                         "shift": np.ones(len(events), dtype=np.int32) * self.global_shift_inst.id,
#                         "weight": weight,
#                     }
#                     for variable_inst in variable_insts:
#                         # prepare the expression
#                         expr = variable_inst.expression
#                         if isinstance(expr, str):
#                             route = Route(expr)
#                             def expr(events, *args, **kwargs):
#                                 if len(events) == 0 and not has_ak_column(events, route):
#                                     return empty_f32
#                                 return route.apply(events, null_value=variable_inst.null_value)
#                         # apply it
#                         fill_data[variable_inst.name] = expr(masked_events)

#                     # fill it
#                     fill_hist(
#                         histograms[var_key],
#                         fill_data,
#                         last_edge_inclusive=self.last_edge_inclusive,
#                     )

#         # merge output files
#         self.output()["hists"].dump(histograms, formatter="pickle")


# # overwrite class defaults
# check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
# CreateHistograms.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
#     default=CreateHistograms.task_family in check_overlap_tasks,
#     add_default_to_description=True,
# )


# CreateHistogramsWrapper = wrapper_factory(
#     base_cls=AnalysisTask,
#     require_cls=CreateHistograms,
#     enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
# )



