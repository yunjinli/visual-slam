
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(argv):
    with open(argv[0], 'rb') as f:
        data = json.load(f)
        cam = data["value0"]
        lm = data["value1"]
        est = data["value2"]
        gt = data["value3"]
        ATE = data["value4"]

        # print((cam[0]["value"]["c.T_w_c"])) ## {'px': 0.0, 'py': 0.0, 'pz': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}, dict
        # print((lm[0]["value"]["lm.p"])) ## {'value0': 0.4959728677267222, 'value1': -3.5238914307859535, 'value2': 11.6504457242374}, dict
        # print(est[0]) ## {'value0': 0.06927333357894985, 'value1': -0.015747674791908035, 'value2': -0.003687501482400273}, dict
        # print(gt[0]) ## {'value0': 0.6932614691264556, 'value1': 0.060777112123096444, 'value2': 0.8411686537939025}, dict

        # print(len(lm))
        # print(len(est))
        # print(len(gt))

        fig1 = plt.figure(figsize=(10,10))
        fig2 = plt.figure(figsize=(10,10))

        ax = fig1.add_subplot(projection='3d')
        ax_2d = fig2.add_subplot()

        gt_x = np.zeros(len(gt))
        gt_y = np.zeros(len(gt))
        gt_z = np.zeros(len(gt))
        for i, p in enumerate(gt):
            gt_x[i] = p['value0']
            gt_y[i] = p['value1']
            gt_z[i] = p['value2']

        est_x = np.zeros(len(est))
        est_y = np.zeros(len(est))
        est_z = np.zeros(len(est))
        for i, p in enumerate(est):
            est_x[i] = p['value0']
            est_y[i] = p['value1']
            est_z[i] = p['value2']

        lm_num = 0
        for i, p in enumerate(lm):
            if p["value"]["lm.p"]['value0']**2 + p["value"]["lm.p"]['value1']**2 + p["value"]["lm.p"]['value2']**2 < 100**2:
                lm_num += 1

        map_x = np.zeros(lm_num)
        map_y = np.zeros(lm_num)
        map_z = np.zeros(lm_num)
        counter = 0
        for i, p in enumerate(lm):
            if p["value"]["lm.p"]['value0']**2 + p["value"]["lm.p"]['value1']**2 + p["value"]["lm.p"]['value2']**2 < 100**2:
                map_x[counter] = p["value"]["lm.p"]['value0']
                map_y[counter] = p["value"]["lm.p"]['value1']
                map_z[counter] = p["value"]["lm.p"]['value2']
                counter += 1

        # for i in est:
        #     i['value0']

        ax_2d.plot(est_x, est_y, c='green', label="Estimated Trajectory")
        ax_2d.plot(gt_x, gt_y, c='red', label="Groun-True Trajectory")
        ax_2d.legend(loc="upper left")
        ax_2d.set_xlabel('X Label')
        ax_2d.set_ylabel('Y Label')
        ax_2d.set_title(f"ATE = {ATE:.3f}")
        ax.plot(est_x, est_y, est_z, c='green')
        ax.plot(gt_x, gt_y, gt_z, c='red')
        ax.axes.set_xlim3d(left=-10, right=10) 
        ax.axes.set_ylim3d(bottom=-10, top=10) 
        ax.axes.set_zlim3d(bottom=-10, top=10) 
        # ax.set_xbound(-5, 5)
        # ax.set_ybound(-5, 5)
        ax.scatter(map_x, map_y, map_z, s=0.5, marker=".", c='black')
        ax.view_init(elev=-120., azim=-90)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.grid(False)
        ax.axis("off")
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
