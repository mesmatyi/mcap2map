# mcap2map

One command from a vehicle MCAP bag to an Autoware-ready map set, with no manual steps:

* `pointcloud_map.pcd` (0.1 m), a 0.2 m copy and 100 m tiles with Autoware metadata, moving objects removed
* `pointcloud_map_rgb.pcd`, the same map coloured from the cameras
* `lanelet2_map.osm`, a Lanelet2 map of the driven roads, checked with Autoware's validator and a routing test
* `calibration.json`, `trajectory_base_link_map_frame.csv`, `map_projector_info.yaml`, validation renders and a
  generated `README.md` with the run's own quality figures

It was built from the Baross híd (Győr) and kek_2 (Kaposvár) map projects and targets the Lexus3 sensor set:
NovAtel INS (BESTUTM + NavSatFix + Odometry), an Ouster-style lidar with per-point time stamps, and cameras with
CameraInfo, all described in `/tf_static`.

## Usage

```bash
./mcap2map.sh run /path/to/drive.mcap
```

The run directory defaults to `<bag stem>_maps` next to the bag (`--out DIR` to change it). The deliverables
land in `DIR/output/`. Other commands:

```bash
./mcap2map.sh batch /path/to/bags/        # every *.mcap in a folder; finished runs resume, failures don't stop it
./mcap2map.sh inspect drive.mcap          # what would be detected (topics, sensors, profile)
./mcap2map.sh status DIR                  # which stages are done
./mcap2map.sh run drive.mcap --from map   # redo a stage and everything after it
./mcap2map.sh run drive.mcap --only checks
```

Bags it cannot map stop at `inspect` with the reason: a damaged MCAP index (repair with `mcap recover`), no
NovAtel INS topics, or no lidar.

A run resumes where it stopped: every stage leaves `DIR/stages/<stage>.json`, and a stage only re-runs when it
is missing or an earlier stage ran after it. Logs are in `DIR/logs/<stage>.log` and `DIR/pipeline.log`.

Optional overrides (`--config cfg.json`, every key optional):

```json
{
  "frame": {"name": "map_kaposvar_0", "ox": 714600, "oy": 5137000},
  "topics": {"points": "/lexus3/os_center/points"},
  "cameras_exclude": ["c2_camera_right_link"],
  "profile": false,
  "params": {"image_stride": 2, "lanelet2": false, "validator_ws": "/path/to/validator_ws",
             "rtk_good_std_m": 0.10, "lo_window_min_s": 5.0, "refine_iterations": 4}
}
```

## Stages and the decisions they make on their own

| stage | what it does | automatic decisions |
|---|---|---|
| `inspect` | reads the bag summary, `/tf_static` and first messages | GNSS/INS topics; the main lidar (most points per scan); cameras = image topics with a CameraInfo, raw or compressed, rectified (pinhole `P`) or not (plumb_bob); base frame; base → lidar / camera / ground extrinsics; vehicle profile |
| `extract` | parallel pass over the bag (time slices), lidar scans to memory-mapped files, frames to JPEG | worker count from free RAM; falls back to one streaming pass if the MCAP index is damaged |
| `frame` | local map frame | reuses a vehicle map frame from `/tf_static` within 10 km, else a round UTM point south-west of the route; UTM zone and geoid undulation from BESTUTM |
| `trajectory` | 1 kHz base_link trajectory from BESTUTM + INS attitude | re-times the INS stream to the GPS epoch grid; antenna lever arm from INS epochs (profile prior as fallback); INS heading offset from course over ground |
| `lidar_calib` | lidar rotation + time offset by scan-to-scan consistency on RTK stretches | starts from the profile prior, or from `/tf_static` turned by the INS heading offset |
| `reference` | pose-graph reference | detects degraded-GNSS windows (not RTK-fixed with std < 10 cm for ≥ 5 s) and replaces the receiver solution there with lidar odometry |
| `refine` | linearised pose graph on smooth corrections | GNSS priors by solution type; lidar submaps 10 m / 25 m apart and between passes; continuity across receiver jumps; residual lidar tilt |
| `map` | deskewed accumulation, rasters, moving-object removal, PCD export | |
| `camera_calib` | camera rotation + time offset by photo-consistency | wide time-offset scan (±1.5 s: camera header stamps can lag by 0.7 s), pinhole vs plumb_bob by held-out consistency |
| `colorize` | colours from keyframes every 1.2 m, depth buffer with the live scan | worker count from free RAM |
| `osm` | Overpass download, map matching, road chains | whole roundabouts mapped, arcs merged into one ring |
| `lanelet` | lane model, PCD paint fit, Lanelet2 assembly | unpainted pieces and pieces whose fit misses the driven path are centred on the driven path |
| `checks` | Autoware validator (if a workspace is configured), routing test, paint precision, smoothness, kinks, connectors, renders | close-up areas from the roundabouts and junction clusters |
| `export` | calibration, trajectory CSV, projector, README | |

## Requirements

* Python 3.12 with `requirements.txt` (`python -m venv .venv && .venv/bin/pip install -r requirements.txt`).
  `mcap2map.sh` uses `./.venv` if present, else the Baross project's venv, or `$MCAP2MAP_PYTHON`.
* Network access for the OpenStreetMap download (the `osm` stage; set `params.lanelet2=false` to skip Lanelet2).
* Optional: ROS 2 Jazzy plus a built `autoware_lanelet2_map_validator` workspace (`params.validator_ws`).
* Disk: about 20 GB per hour of raw-camera bag for extracted frames and scans (`params.image_stride` reduces
  it); RAM: 16 GB is enough (stages run as separate processes).

## Vehicle profiles

`mcap2map/profiles/*.json` hold calibration priors for a known vehicle (matched by base and lidar frame
names): lidar correction, antenna lever arm, camera extrinsics and stamp offsets. Every run re-estimates
them; the prior only gives the optimisers a good start. `lexus3.json` comes from the kek_2 bag.

## Safety nets in the Lanelet2 stage

* Pieces without paint, and pieces whose paint fit leaves the driven path outside its lane, are re-modelled from
  the driven path. Seams between the two kinds are blended so successor links survive.
* Successor links broken by small gaps (≤ 2 m) are stitched; degenerate lanelets (< 1 m) are dropped.
* Every manoeuvre the vehicle drove must be routable. Where a driven transition has no route (missing junction
  connector, lane change before a turn), a connector is added and tagged `mcap2map:driven_manoeuvre=yes` so you can
  review it.
* A one-lane roundabout ring takes only the rightmost lane of a multi-lane arm; the other lanes reach it by a lane
  change.
* Junction nodes joined by an OSM chain shorter than 12 m (for example a short stem between a junction and a
  roundabout flare) are treated as one junction, so the connectors span the whole area.
* Where OSM draws a side road running along a through road for more than 15 m before their shared node, the
  junction is moved to where the side road leaves the through road: the side road is cut there and the through road
  is split into a new junction (`junction relocation` in `logs/lanelet.log`).
* Where OSM draws a roundabout smaller than the measured ring, so that a flared arm's split node lies at or inside
  the measured outer kerb, the arm is rebuilt. The approach road splits 10 m before the kerb, and the entry and exit
  links follow the paths the vehicle drove (`arm rebuilt` in the log).

## Known limits

* The trajectory needs a NovAtel INS (BESTUTM, NavSatFix, Odometry). Bags with another receiver stop at `inspect`
  with a message.
* In long GNSS-denied stretches the map stays locally sharp, but its absolute position rests on lidar odometry
  anchored at the stretch ends.
* Lanes on unpainted streets follow the driven path at 3.5 m width; traffic-light heads are placeholders.
* OSM geometry errors that the safety nets above do not catch still reach the map. The route test and the
  validator report them. The repairs themselves have limits:
  * Turn restrictions that name a way merged into a junction cluster or a rebuilt roundabout link are dropped. A
    relocated junction gets no OSM turn restrictions.
  * A rebuilt roundabout arm assumes the splitter island starts 10 m before the kerb. A link the vehicle did not
    drive continues the approach lane in a straight line.
* The Lanelet2 stage is heuristic: small changes in the trajectory can move a few pieces between the paint-fitted
  and driven-path models. Check `output/README.md` and `validation/` after every run.
* Only the driven roads are mapped.

Unattended operation on new recordings: run `./mcap2map.sh batch /path/to/bags/` from cron or a systemd timer.
Finished bags are skipped (their stages are up to date) and new ones get mapped.

## Layout

```
mcap2map/            package: CLI (__main__), pipeline (stages), inspect_bag, report
mcap2map/tools/      the stage tools (standalone scripts run with MCAP2MAP_RUN=<run.json>)
mcap2map/profiles/   vehicle calibration priors
mcap2map.sh          wrapper
```
