
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


class CreateFakeFactorHistograms(
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
                
                h = (hist.Hist.new
                    .IntCat([], name="category", growth=True)
                    .IntCat([], name="process", growth=True)
                    .IntCat([], name="shift", growth=True))
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
                    "shift"             : np.ones(len(events), dtype=np.int32) * self.global_shift_inst.id,
                    "weight": weight,
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
CreateFakeFactorHistograms.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=CreateFakeFactorHistograms.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


CreateFakeFactorHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateFakeFactorHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)

class MergeFakeFactors(
    VariablesMixin,
    DatasetsProcessesMixin,
    CategoriesMixin,
    WeightProducerMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
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
        CreateFakeFactorHistograms=CreateFakeFactorHistograms,
    )
    
    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets" )#, f"datasets_{self.datasets_repr}")
        return parts
    
    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name})
            for cat_name in sorted(self.categories)
        ]

    def _get_variables(self):
        if self.is_workflow():
            return self.as_branch()._get_variables()

        variables = self.variables

        # optional dynamic behavior: determine not yet created variables and require only those
        if self.only_missing:
            missing = self.output().count(existing=False, keys=True)[1]
            variables = sorted(missing, key=variables.index)

        return variables

    def workflow_requires(self):
        reqs = super().workflow_requires()
        if not self.pilot:
            variables = self._get_variables()
            if variables:
                reqs["ff_method"] = self.reqs.CreateFakeFactorHistograms.req_different_branching(
                    self,
                    branch=-1,
                    variables=tuple(variables),
                )

        return reqs

    def requires(self):
        return {
            d: self.reqs.CreateFakeFactorHistograms.req(
                self,
                dataset=d,
                branch=-1,
            )
            for d in self.datasets
        }
    def output(self):
        return {"hists": self.target(f"fake_factors.pickle")}

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        import matplotlib.pyplot as plt
        # preare inputs and outputs
        inputs = self.input()
        outputs = self.output()
        merged_per_dataset = {}
        projected_hists = []
        for (dataset_name, dataset) in inputs.items():
            files = dataset['collection']
            # load input histograms per dataset
            hists = [
                inp['hists'].load(formatter="pickle")['fake_factors']
                for inp in self.iter_progress(files.targets.values(), len(files), reach=(0, 50))
            ]
            self.publish_message(f"merging Fake factor histograms for {dataset_name}")
            the_hist = sum(hists[1:], hists[0].copy())
            merged_per_dataset[dataset_name] = the_hist
            #Get axes names excluding 'process'. This is needed to merge hists for different processes
            ax_names = [ax_name for ax_name in the_hist.axes.name if ax_name != 'process']
            #Remove 'process' axis by projecting hist on the remaining axes
            projected_hists.append(the_hist.project(*ax_names))
        merged_hist = sum(projected_hists[1:], projected_hists[0].copy())
        
        cat_SR = self.config_inst.get_category(self.branch_data.category)
        cat_DR_den = self.config_inst.get_category(cat_SR.x.DR_den)
        cat_DR_num = self.config_inst.get_category(cat_SR.x.DR_num)
        
        def get_hist (h, category): 
            return h[{"category": hist.loc(category.id)}]
        
        h_DR_num = get_hist(merged_hist,cat_DR_num).values()
        h_DR_den = get_hist(merged_hist,cat_DR_den).values()
        
        ff_values = np.where((h_DR_num > 0) & (h_DR_den > 0),
                             h_DR_num / np.maximum(h_DR_den, 1),
                             0.0,
        )
        
        #For the control: make 2d hists and plot them:
        hist2d = merged_hist.project('tau_pt','tau_dm_pnet')
        ff_hist = hist.Hist(*hist2d.axes, data=ff_values[0])
        fig, ax = plt.subplots(figsize=(12, 8))
        ff_hist.plot2d(ax=ax)
        plt.savefig('fake_factors.pdf')
        from IPython import embed; embed()
        #outputs["hists"][variable_name].dump(merged, formatter="pickle")F

        # optionally remove inputs
        if self.remove_previous:
            inputs.remove()


# MergeFakeFactorsWrapper = wrapper_factory(
#     base_cls=AnalysisTask,
#     require_cls=MergeFakeFactors,
#     enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
# )

