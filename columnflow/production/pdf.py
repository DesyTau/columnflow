# coding: utf-8

"""
Column production methods related to the PDF weights.
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

import law

ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    uses={"LHEPdfWeight"},
    produces={
        "pdf_weight", "pdf_weight_up", "pdf_weight_down",
    },
)
def pdf_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that determines the pdf up and down variations on an event-by-event basis.
    This producer assumes that the nominal entry is always the first LHEPdfWeight value and
    that the nominal weight is already included in the LHEWeight.
    Can only be called with MC datasets.

    Resources:

       - https://arxiv.org/pdf/1510.03865.pdf
    """
    # stop here for data
    if self.dataset_inst.is_data:
        raise ValueError("attempt to determine pdf variations in data")

    # check for the correct amount of weights
    n_weights = ak.num(events.LHEPdfWeight, axis=1)
    bad_mask = (n_weights != 101) & (n_weights != 103)
    if ak.any(bad_mask):
        bad_values = set(n_weights[bad_mask])
        raise Exception(
            "the number of LHEPdfWeights is expected to be 101 or 103, but also found values " +
            f"{bad_values} in dataset {self.dataset_inst.name}",
        )

    # TODO: logger if nominal != 1
    if ak.any(events.LHEPdfWeight[:, 0] != 1):
        bad_values = set(events.LHEPdfWeight[:, 0][events.LHEPdfWeight[:, 0] != 1])
        logger.debug(
            "The nominal LHEPdfWeight is expected to be 1 but also found values " +
            f"{bad_values} in dataset {self.dataset_inst.name}. All variations will be " +
            "normalized to the nominal LHEPdfWeight and it is assumed that the nominal " +
            "weight is already included in the LHEWeight.",
        )

    # normalize all weights by the nominal one, assumed to be the first value
    pdf_weights = events.LHEPdfWeight[:, 1:101] / events.LHEPdfWeight[:, 0]
    pdf_weights = ak.sort(pdf_weights, axis=1)

    # PDF uncertainty as half the width of the central 68% CL
    stddev = (pdf_weights[:, 83] - pdf_weights[:, 15]) / 2

    # store columns
    events = set_ak_column(events, "pdf_weight", ak.ones_like(events.event))
    events = set_ak_column(events, "pdf_weight_up", 1 + stddev)
    events = set_ak_column(events, "pdf_weight_down", 1 - stddev)

    return events
