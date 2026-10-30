// -*- C++ -*-
//
// Copyright (C) 2020-2025 MACESW developers
//
// This file is part of MACESW, Muonium-to-Antimuonium Conversion Experiment
// offline software.
//
// MACESW is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// MACESW is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// MACESW. If not, see <https://www.gnu.org/licenses/>.

// Optional geometry exporter for the Phase-I graph reconstruction study.

#include "MACE/Detector/Description/ECAL.h++"
#include "MACE/PhaseI/Detector/Description/SciFiTracker.h++"
#include "MACE/PhaseI/Detector/Description/TTC.h++"
#include "MACE/PhaseI/Detector/Description/UsePhaseIDefault.h++"
#include "MACE/PhaseI/SimMACEPhaseI/Action/DetectorConstruction.h++"

#include "Mustard/Detector/Description/DescriptionIO.h++"
#include "Mustard/Env/BasicEnv.h++"

#include "fmt/format.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<typename P>
auto Print3(const P& p) -> void {
    fmt::print("[{:.17g},{:.17g},{:.17g}]", p.x(), p.y(), p.z());
}

auto Export(const std::filesystem::path& yaml) -> void {
    using Description = MACE::PhaseI::SimMACEPhaseI::Action::DetectorConstruction::ProminentDescription;
    MACE::PhaseI::Detector::Description::UsePhaseIDefault();
    (void) MACE::PhaseI::Detector::Description::SciFiTracker::Instance();
    (void) MACE::PhaseI::Detector::Description::TTC::Instance();
    Mustard::Detector::Description::DescriptionIO::Import<Description>(yaml);

    const auto& scifi{MACE::PhaseI::Detector::Description::SciFiTracker::Instance()};
    const auto& ttc{MACE::PhaseI::Detector::Description::TTC::Instance()};
    const auto& ecal{MACE::Detector::Description::ECAL::Instance()};
    const auto& fibers{scifi.DetectorFiberInformation()};
    const auto& modules{ecal.Array().moduleList};

    if (fibers.size() != 2360 or ttc.NTile() != 108 or modules.size() != 488) {
        throw std::runtime_error(fmt::format("unexpected Phase-I channel counts: SciFi {}, TTC {}, ECAL {}",
                                             fibers.size(), ttc.NTile(), modules.size()));
    }

    fmt::print("{{\"schema_version\":1,\"units\":\"mm\",\"coordinate_frame\":\"MACE world\",\"detectors\":{{\"scifi\":{{\"channels\":[");
    for (std::size_t i{}; i < fibers.size(); ++i) {
        const auto& f{fibers[i]};
        if (i) fmt::print(",");
        fmt::print("{{\"id\":{},\"layer_id\":{},\"local_id\":{},\"layer_type\":\"{}\",\"radius_mm\":{:.17g},\"pitch_rad\":{:.17g},\"rotation_angle_rad\":{:.17g}}}",
                   i, f.layerID, f.localID, f.layerType, f.radius, f.pitch, f.rotationAngle);
    }
    fmt::print("]}},\"ttc\":{{\"channels\":[");
    for (int id{}; id < ttc.NTile(); ++id) {
        if (id) fmt::print(",");
        fmt::print("{{\"id\":{},\"position_mm\":", id);
        Print3(ttc.TilePosition(id));
        fmt::print(",\"normal\":");
        Print3(ttc.TileNormal(id));
        fmt::print("}}");
    }
    fmt::print("]}},\"ecal\":{{\"channels\":[");
    for (std::size_t i{}; i < modules.size(); ++i) {
        const auto& m{modules[i]};
        if (i) fmt::print(",");
        fmt::print("{{\"id\":{},\"position_mm\":", m.moduleID);
        Print3(m.centroid);
        fmt::print(",\"normal\":");
        Print3(m.normal);
        std::vector<int> neighbors;
        neighbors.reserve(m.neighborModuleID.size());
        for (const auto n : m.neighborModuleID) neighbors.push_back(n);
        std::ranges::sort(neighbors);
        fmt::print(",\"neighbors\":[");
        for (std::size_t j{}; j < neighbors.size(); ++j) {
            if (j) fmt::print(",");
            fmt::print("{}", neighbors[j]);
        }
        fmt::print("]}}");
    }
    fmt::print("]}}}}}}\n");
}

} // namespace

auto main(int argc, char** argv) -> int {
    if (argc != 2) {
        std::cerr << "usage: phasei_gnn_geometry_export <SimMACEPhaseI_geom.yaml>\n";
        return 2;
    }
    try {
        Mustard::Env::BasicEnv env{argc, argv, {}, {}, false};
        Export(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "geometry export failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
