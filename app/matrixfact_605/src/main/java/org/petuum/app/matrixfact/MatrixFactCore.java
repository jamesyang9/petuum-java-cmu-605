package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;

import org.petuum.ps.PsTableGroup;
import org.petuum.ps.row.double_.DenseDoubleRow;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.petuum.ps.common.util.Timer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class MatrixFactCore {
    private static final Logger logger =
            LoggerFactory.getLogger(MatrixFactCore.class);

    // Perform a single SGD on a rating and update LTable and RTable
    // accordingly.
    public static void sgdOneRating(Rating r, double learningRate,
                                    DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
        int i = r.userId;
        int j = r.prodId;
        float v = r.rating;

        DenseRow L_i = (DenseRow) (LTable.get(i));
        DenseRow R_j = (DenseRow) (RTable.get(j));

        DoubleRowUpdate lUpdates = new DenseDoubleRowUpdate(k);
        DoubleRowUpdate rUpdates = new DenseDoubleRowUpdate(k);

        double dotProduct = 0;
        for (int k = 0; k < K; k++) {
            dotProduct += L_i.get(k) * R_j.get(k);
        }

        for (int k = 0; k < K; k++) {
            double deltaL = -2.0 * (v - dotProduct) * R_j.get(k) + 2.0 * lambda * L_i.get(k);
            double deltaR = -2.0 * (v - dotProduct) * L_i.get(k) + 2.0 * lambda * R_j.get(k);
            lUpdates.setUpdate(k, deltaL);
            rUpdates.setUpdate(k, deltaR);
        }

        LTable.batchInc(i, lUpdates);
        RTable.batchInc(j, rUpdates);
    }

    // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
    // row [LRowBegin, LRowEnd) of LTable,  [RRowBegin, RRowEnd) of Rtable.
    // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
    // lossRecorder.
    public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
                                    int elemBegin, int elemEnd, DoubleTable LTable,
                                    DoubleTable RTable, int LRowBegin, int LRowEnd, int RRowBegin,
                                    int RRowEnd, LossRecorder lossRecorder, int K, double lambda) {
        double sqLoss = 0;
        double totalLoss = 0;

        for (Rating r: ratings) {
            int i = r.userId;
            int j = r.prodId;
            float v = r.rating;
            DenseRow L_i = (DenseRow) (LTable.get(i));
            DenseRow R_j = (DenseRow) (RTable.get(j));
            double dotProduct = 0;
            for (int k = 0; k < K; k++) {
                dotProduct += L_i.get(k) * R_j.get(k);
            }
            sqLoss += Math.pow((v - dotProduct), 2);
        }

        totalLoss = sqLoss;
        for (int i = LRowBegin; i < LRowEnd; i++) {
            DenseRow L_i = (DenseRow) (LTable.get(i));
            for (int k = 0; k < K; k++) {
                totalLoss += lambda * Math.pow(L_i.get(k), 2);
            }
        }
        for (int j = RRowBegin; j < RRowEnd; j++) {
            DenseRow R_j = (DenseRow) (RTable.get(j));
            for (int k = 0; k < K; k++) {
                totalLoss += lambda * Math.pow(R_j.get(k), 2);
            }
        }

        lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
        lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
        lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
    }
}
