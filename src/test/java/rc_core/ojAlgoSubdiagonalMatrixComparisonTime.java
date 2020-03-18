package rc_core;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.ojalgo.function.NullaryFunction;
import org.ojalgo.function.UnaryFunction;
import org.ojalgo.matrix.Primitive64Matrix;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.type.CalendarDateUnit;
import org.ojalgo.type.Stopwatch;

import java.util.Random;

/**
 * Speed comparison of various (subdiagonal) matrix implementations.
 * 
 * The custom implementation creates cycles -- different.
 */
public class ojAlgoSubdiagonalMatrixComparisonTime {
    private static double valueW;
    private static final Stopwatch stopwatch = new Stopwatch();
    private static final int N_x = 5;
    
    @BeforeAll
    private static void setup() {
        Random random = new Random();
        valueW = random.nextDouble();
    }
    
    @Test
    /*4.*/
    void densePrimitiveMatrix() {
        stopwatch.reset();
        /* Primitive64Matrix (Dense) */
        Primitive64Matrix.Factory matrixFactory = Primitive64Matrix.FACTORY;
        NullaryFunction<Double> cyclicReservoirWithJumps = new NullaryFunction<Double>() {
            @Override
            public double doubleValue() {
                return valueW;
            }

            @Override
            public Double invoke() {
                return valueW;
            }
        };
        Primitive64Matrix.LogicalBuilder matrixBuilder = matrixFactory.makeFilled(N_x, N_x, cyclicReservoirWithJumps).logical();
        Primitive64Matrix identity = Primitive64Matrix.FACTORY.makeEye(N_x, N_x);
        Primitive64Matrix W_internal = matrixBuilder.bidiagonal(false).get().subtract(identity.multiply(valueW)); // attempt for subdiagonal matrix creation
        System.out.println("time: " + stopwatch.stop(CalendarDateUnit.MICROS));
//        System.out.println("dense version: " + W_internal);
    }
    
    @Test
    /*3.*/
    void sparsePrimitiveMatrix() {
        stopwatch.reset();
        /* Primitive64Matrix (Sparse) */
        Primitive64Matrix.Factory matrixFactory = Primitive64Matrix.FACTORY;
        Primitive64Matrix.SparseReceiver W_sparse_receiver = matrixFactory.makeSparse(N_x, N_x);
        W_sparse_receiver.loopAll((i, j) -> {if (i-1 == j) W_sparse_receiver.add(i, j, valueW);});
        System.out.println("time: " + stopwatch.stop(CalendarDateUnit.MICROS));
//        System.out.println("sparse version: " + W_sparse_receiver.get());
    }
    
    @Test
    /*5.*/
    void sparseStore() {
        stopwatch.reset();
        /* SparseStore */
        SparseStore<Double> W_sparse_store = SparseStore.PRIMITIVE64.make(N_x, N_x);
        W_sparse_store.modifyAll(new UnaryFunction<Double>() {
            @Override
            public double invoke(double arg) {
                return valueW;
            }

            @Override
            public float invoke(float arg) {
                return (float) valueW;
            }

            @Override
            public Double invoke(Double arg) {
                return valueW;
            }
        });
        MatrixStore.LogicalBuilder<Double> storeBuilder  = W_sparse_store.logical();
        MatrixStore<Double> W_internal_store = storeBuilder.bidiagonal(false).get();
        SparseStore<Double> identityStore = SparseStore.makePrimitive(N_x, N_x);
        identityStore.fillDiagonal(1.0);
        W_internal_store = W_internal_store.subtract(identityStore.multiply(valueW));
        System.out.println("time: " + stopwatch.stop(CalendarDateUnit.MICROS));
//        System.out.println("sparse store version: " + W_internal_store);
    }
    
    @Test
    /*6.*/
    void sparseStore2() {
        stopwatch.reset();
        /* Sparse Matrix */
        MatrixStore<Double> W_internal_store = SparseStore.makePrimitive(N_x, N_x);
        W_internal_store = W_internal_store.operateOnAll(new UnaryFunction<Double>() {
            @Override
            public double invoke(double arg) {
                return valueW;
            }

            @Override
            public float invoke(float arg) {
                return (float) valueW;
            }

            @Override
            public Double invoke(Double arg) {
                return valueW;
            }
        });
        W_internal_store = W_internal_store.logical().bidiagonal(false).get();
        SparseStore<Double> identitySparse = MatrixStore.PRIMITIVE64.makeSparse(N_x, N_x);
//        SparseStore<Double> identitySparse2 = SparseStore.PRIMITIVE64.make(N_x, N_x);
        identitySparse.fillDiagonal(1.0);
        W_internal_store = W_internal_store.subtract(identitySparse.multiply(valueW));
        System.out.println("time: " + stopwatch.stop(CalendarDateUnit.MICROS));
//        System.out.println("sparse store (2): " + W_internal_store);
    }
    
    @Test
    /*2.*/
    void sparseStoreQuick() {
        stopwatch.reset();
        /* SparseStore Quick */
        SparseStore<Double> W_internal_sparse_alt = SparseStore.makePrimitive(N_x, N_x);
        W_internal_sparse_alt.loopAll((x, y) -> {if(x-1 == y) W_internal_sparse_alt.add(x, y, valueW);});
        System.out.println("time " + stopwatch.stop(CalendarDateUnit.MICROS));
//        System.out.println("sparse quick version: " + W_internal_sparse_alt);
    }
    
    @Test
    /*1.*/
    void sparseStoreQuicker() {
        stopwatch.reset();
        /* SparseStore Quicker */
        SparseStore<Double> W_internal_sparse_alt2 = SparseStore.makePrimitive(N_x, N_x);
        for (int i = 0; i < N_x; ++i) {
            W_internal_sparse_alt2.add(i, i-1, valueW);
        }
        System.out.println("time " + stopwatch.stop(CalendarDateUnit.MICROS));
//        System.out.println("sparse quicker version: " + W_internal_sparse_alt2);
    }
    
    @Test
    void customMatrixStore() {
        stopwatch.reset();
        /* Custom MatrixStore */
        JumpsSaturatedMatrix W_input_jumps = new JumpsSaturatedMatrix(N_x, 1, 3);
        System.out.println("time: " + stopwatch.stop(CalendarDateUnit.MICROS));
        System.out.println("custom store w/ jumps: " + W_input_jumps);
    }
}
