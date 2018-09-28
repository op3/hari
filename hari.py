#!/usr/bin/env python3

# Copyright 2017 Udo Gayer (gayer.udo@gmail.com)
# Copyright 2018 Oliver Papst (opapst@ikp.tu-darmstadt.de)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# ithe Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import sys
import logging
from pathlib import Path
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import read_input
import binning


_log = logging.getLogger("hari")


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


class MismatchingBinCountException(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog="hari",
        description="hari: histogram arbitrary rebinning intelligently")
    input_bins_group = parser.add_mutually_exclusive_group()
    output_bins_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "histogram",
        metavar=("HISTOGRAM_FILE"),
        type=Path,
        help="histogram file (single- or two-column) or TH1 inside ROOT-file")
    input_bins_group.add_argument(
        "-ib", "--input_bins",
        metavar=("BIN_FILE"),
        type=Path,
        help="Single-column file that contains the bin centers of the input")
    output_bins_group.add_argument(
        "-ob", "--output_bins",
        metavar=("BIN_FILE"),
        type=Path,
        help="Single-column file that contains the bin centers of the output")
    input_bins_group.add_argument(
        "-c", "--calibration",
        metavar=("CALIBRATION_FILE"),
        type=Path,
        help="Use file with calibration parameters for polynomial")
    parser.add_argument(
        "-d", "--deterministic",
        help="Rebin deterministically",
        action="store_true")
    output_bins_group.add_argument(
        "-f", "--binning_factor", type=float, help="Rebinning factor")
    parser.add_argument(
        "-i", "--conserve_integral",
        help="Conserve sum over h[i]*dx[i] instead of sum over h[i]",
        action="store_true")
    parser.add_argument(
        "-k", "--spline_order",
        help="Order of the spline interpolation (default: k == 3)",
        default=3,
        type=int,
        metavar="[1-5]")
    parser.add_argument(
        "-l", "--limits",
        nargs=2,
        metavar=("LOWER_LIMIT", "UPPER_LIMIT"),
        type=float,
        action=Store_as_array,
        help="Set limits of the plot range")
    output_bins_group.add_argument(
        "-n", "--n_bins",
        type=int,
        help="Number of bins for the output histogram")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Set output file name")
    parser.add_argument(
        "-p", "--plot",
        help="Create plots of the rebinned spectrum",
        action="store_true")
    parser.add_argument(
        "-r", "--range",
        nargs=2,
        metavar=("START", "STOP"),
        type=float,
        action=Store_as_array,
        help="Set range of the output bins")
    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="Set random number seed")
    parser.add_argument(
        "-R", "--root",
        type=str,
        metavar="HISTOGRAM_NAME",
        help=argparse.SUPPRESS)
        #help="Specify root histogram name and output to root file instead")
    parser.add_argument(
        "-v", "--verbose",
        help="Print messages during program execution",
        action="store_true")
    args = parser.parse_args()


    _log.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    if args.verbose:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('‣ %(levelname)s – %(message)s')
    ch.setFormatter(formatter)
    _log.addHandler(ch)

    if args.seed:
        np.random.seed(args.seed)

    try:
        input_bins, input_hist = read_input_histogram(
            args.histogram, args.input_bins, args.calibration)

        n_input_bins = np.shape(input_bins)[-1]
        _log.info(f"Input Histogram: {n_input_bins} bins "
                  f"from {input_bins.min()} to {input_bins.max()}")

        output_bins = get_output_bins(
            input_bins, args.range, args.output_bins,
            args.binning_factor, args.n_bins)

        n_output_bins = np.size(output_bins)

        _log.info(f"Output Histogram: {n_output_bins} bins "
                  f"from {output_bins[0]} to {output_bins[-1]}")

        output_hist, input_bins_width, output_bins_width = rebin(
            input_bins, input_hist, output_bins,
            args.spline_order, args.conserve_integral, args.deterministic)

        # Calculate calibration for output histogram
        if args.binning_factor or args.n_bins:
            calibration_coefficients = np.array([
                output_bins[0],
                (output_bins[1] - output_bins[0]) / n_output_bins])

        # Write output file
        if not args.root:
            if args.output:
                output_hist_filename = args.output.with_suffix(".hist.txt")
                output_bins_filename = args.output.with_suffix(".bins.txt")
            else:
                output_hist_filename = args.histogram.with_suffix(".hist.txt")
                output_bins_filename = args.histogram.with_suffix(".bins.txt")

            _log.info(f"Writing output histogram to {output_hist_filename}")
            np.savetxt(output_hist_filename, output_hist, fmt='%.1f')
            _log.info(f"Writing output bins to {output_bins_filename}")
            np.savetxt(output_bins_filename, output_bins, fmt='%.6e')

            if args.binning_factor or args.n_bins:
                if args.output:
                    output_cal_filename = args.output.with_suffix(".cal.txt")
                else:
                    output_cal_filename = args.histogram.with_suffix(".cal.txt")
                _log.info(f"Writing calibration coefficients to {output_cal_filename}")
                with open(output_cal_filename, "w") as output_cal_file:
                    output_cal_file.write(f"{output_hist_filename}:")
                    for c in calibration_coefficients:
                        output_cal_file.write("\t")
                        output_cal_file.write(str(c))
        else:
            _log.error("Output of root files not yet implemented")
            #try:
            #    import ROOT
            #    if args.output:
            #        output_root_filename = args.output
            #    else:
            #        output_root_filename = args.histogram.with_suffix(".root")
            #    output_root_file = ROOT.TFile(output_root_filename, "UPDATE")
            #    
            #    output_root_file.Write()
            #    output_root_file.Close()
            #except BaseException:
            #    pass

    except Exception as err:
        _log.error(err)
        raise err
        #exit(127)

    # Print information
    if args.conserve_integral:
        # Calculate integral over histograms
        input_integral = np.sum(input_hist * input_bins_width)
        output_integral = np.sum(output_hist * output_bins_width)

        _log.info(f"Integral over input histogram: {input_integral}")
        _log.info(f"Integral over output histogram: {output_integral}")
        if args.deterministic:
            _log.info(f"Change of total histogram integral due to interpolation: {(input_integral - output_integral) / input_integral * 100.} %")
        else:
            _log.info(f"Change of total histogram integral due to interpolation and random sampling: {(input_integral - output_integral) / input_integral * 100.} %")

    else:
        # Calculate sum of histogram bins
        n_input = np.sum(input_hist)
        n_output = np.sum(output_hist)

        _log.info(f"Sum of input histogram bins: {n_input}")
        _log.info(f"Sum of output histogram bins: {n_output}")
        if args.deterministic:
            _log.info(f"Change of total histogram content due to interpolation: {(n_output - n_input) / n_input * 100.} %")
        else:
            _log.info(f"Change of total histogram content due to interpolation and random sampling: {(n_output - n_input) / n_input * 100.} %")

    # Plot result
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(2, sharex=True)

            if args.limits:
                energy_range = args.limits
            else:
                energy_range = np.array([np.min(input_bins), np.max(input_bins)])

            ax[0].step(
                np.extract((input_bins > energy_range[0]) * (input_bins < energy_range[1]), input_bins),
                np.extract((input_bins > energy_range[0]) * (input_bins < energy_range[1]), input_hist),
                where="mid", color="black")
            ax[1].step(
                np.extract((output_bins > energy_range[0]) * (output_bins < energy_range[1]), output_bins),
                np.extract((output_bins > energy_range[0]) * (output_bins < energy_range[1]), output_hist),
                where="mid", color="green")

            f.tight_layout()
            f.subplots_adjust(hspace=0.)
            plt.show()
        except ImportError:
            _log.error("Please install matplotlib for plotting")


def rebin(input_bins, input_hist, output_bins, spline_order=3,
         conserve_integral=False, deterministic=False):

    # Calculate the lower and upper limits of the bins
    if np.shape(input_bins)[0] == 2:
        input_bins_low, input_bins_high = input_bins
        input_bins_center = (input_bins_low + input_bins_high)/2
    else:
        input_bins_low, input_bins_high = binning.calc_bin_limits(input_bins)
        input_bins_center = input_bins

    if np.shape(output_bins)[0] == 2:
        output_bins_low, output_bins_high = output_bins
    else:
        output_bins_low, output_bins_high = binning.calc_bin_limits(output_bins)

    n_input_bins = np.size(input_bins_low)
    n_output_bins = np.size(output_bins_low)

    # Calculate bin widths
    input_bins_width = input_bins_high - input_bins_low
    output_bins_width = output_bins_high - output_bins_low

    # Interpolate the input histogram
    if conserve_integral:
        _log.info("Interpolating the input histogram. Conserving the integral over h[i]*dx[i].")
        inter = InterpolatedUnivariateSpline(
            input_bins_center, input_hist, k=spline_order)
    else:
        _log.info("Interpolating the input histogram. Conserving the sum over the bin contents h[i].")
        inter = InterpolatedUnivariateSpline(
            np.arange(0., n_input_bins), input_hist, k=spline_order)

    inter_bins = InterpolatedUnivariateSpline(
        input_bins_center, np.arange(0., n_input_bins), k=spline_order)


    # Find new bins which are outside the original histogram.
    extra_bins = (output_bins_low >= input_bins_low[0]) * (output_bins_high <= input_bins_high[-1])
    if extra_bins.any():
        _log.info(f"New histogram has {n_output_bins - np.sum(extra_bins)} new bins outside the range of the old histogram. They will be filled with zeros.")

    # Calculate the bin contents of the output histogram
    inter_integral_v = np.vectorize(inter.integral)
    if conserve_integral:
        output_hist = np.maximum(inter_integral_v(
            output_bins_low, output_bins_high) /
            output_bins_width, 0.)
    else:
        output_hist = np.maximum(inter_integral_v(inter_bins(
            output_bins_low), inter_bins(output_bins_high)), 0.)

    if not deterministic:
        _log.info("Rebinning and preserving the statistical fluctuations")
        output_hist = np.random.poisson(output_hist)
    else:
        _log.info("Rebinning without preserving the statistical fluctuations")

    return output_hist, input_bins_width, output_bins_width


def read_input_histogram(histogram_path, input_bins_path=False, calibration=False):
    _log.info(f"Reading input histogram from file {histogram_path}")

    try:
        input_hist = np.loadtxt(histogram_path)
    except OSError:
        _log.info("File not found, attempting to read ROOT histogram")
        root_path = []
        while True:
            root_path.append(histogram_path.name)
            histogram_path = histogram_path.parent
            if str(histogram_path) == histogram_path.anchor:
                raise FileNotFoundError()
            if not histogram_path.is_file():
                continue
            try:
                import uproot
                with uproot.open(histogram_path) as hist_file:
                    input_th1 = hist_file["/".join(reversed(root_path))]
                input_hist, x = input_th1.numpy
                input_bins = np.array([x[:-1], x[1:]])
                return input_bins, input_hist

            except ImportError:
                import ROOT
                root_file = ROOT.TFile(str(histogram_path))
                root_hist = root_file.Get("/".join(reversed(root_path)))
                input_hist = np.zeros([root_hist.GetNbinsX() - 1, 2])
                # ROOT File calibration for tv
                try:
                    from root_numpy import hist2array
                    input_hist = hist2array(root_hist)
                except ImportError:
                    for i in range(root_hist.GetNbinsX()):
                        input_hist[i - 1] = [root_hist.GetBinCenter(i), root_hist.GetBinContent(i)]
                break
            except BaseException:
                # Working with PyROOT feels really stupid
                pass


    input_hist_size = np.shape(input_hist)[0]
    
    if len(np.shape(input_hist)) == 2:
        _log.info(f"Reading input bins from file {histogram_path}")
        input_bins = input_hist[:, 0]
        input_hist = input_hist[:, 1]
    elif input_bins_path:
        _log.info(f"Reading input bins from file {input_bins_path}")
        input_bins = np.loadtxt(input_bins_path)
        n_input_bins = np.size(input_bins)
        if n_input_bins != input_hist_size:
            raise MismatchingBinCountException(
                "Number of bins does not match.\n"
                f"\tHistogram size: {input_hist_size}\n"
                f"\tNumber of bins: {n_input_bins}")
    elif calibration:
        _log.info(f"Reading calibration parameters for input histogram from file {calibration}")
        input_bins = read_input.calibrate(
            input_hist_size, calibration,
            histogram_path)
    else:
        _log.info("No input bins given, assume bin center == number of bin")
        input_bins = np.arange(0, input_hist_size)

    return input_bins, input_hist


def get_output_bin_range(a_range, input_bins):
    if a_range is not None:
        return a_range
    else:
        _log.info("No range for output bins given, assume same range as input")
        return np.array([input_bins.min(), input_bins.max()])


def get_output_bins(input_bins, a_range=False, output_bins_path=False, binning_factor=False, n_bins=False):
    n_input_bins = np.shape(input_bins)[-1]
    output_bin_range = get_output_bin_range(a_range, input_bins)
    if output_bin_range[0] < input_bins.min() or output_bin_range[1] < input_bins.max():
        _log.warning(f"Range of output bins [{output_bin_range[0]}, {output_bin_range[1]}] is larger than input bins [{input_bins.min()}, {input_bins.max()}]")

    if binning_factor:
        _log.info(f"Rebinning factor: {binning_factor}")
        output_bins = np.linspace(output_bin_range[0], output_bin_range[1], int(
            n_input_bins / binning_factor))
    elif output_bins_path:
        _log.info(f"Reading output bins from {output_bins_path}")
        output_bins = np.loadtxt(output_bins_path)
        if output_bins[0] < input_bins[0] or output_bins[-1] < input_bins[-1]:
            _log.warning("Range of output bins [" + output_bin_range[0] +
                     ", " +  output_bin_range[1] +
                     "] is larger than input bins [" + input_bins[0] +
                     ", " + input_bins[-1] + "]")
    elif n_bins:
        _log.info(f"Number of output bins set to: {n_bins}")
        output_bins = np.linspace(
            output_bin_range[0], output_bin_range[1], n_bins)
    else:
        _log.info("No output bins given, assume bin center == number of bin")
        output_bins = np.linspace(input_bins[0], input_bins[-1], n_input_bins)
    return output_bins


if __name__ == "__main__":
    main()
