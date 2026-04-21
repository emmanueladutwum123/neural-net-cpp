#include <gtest/gtest.h>
#include "nn/matrix.hpp"
#include "nn/layers.hpp"
#include "nn/activations.hpp"
#include "nn/loss.hpp"
#include "nn/optimizer.hpp"

using namespace nn;

TEST(MatrixTest, Multiply) {
    Matrix a(2, 3);
    Matrix b(3, 2);
    for (int i = 0; i < 6; ++i) { a.data[i] = i + 1.0f; b.data[i] = i + 1.0f; }
    auto c = a.matmul(b);
    ASSERT_EQ(c.rows, 2);
    ASSERT_EQ(c.cols, 2);
    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    EXPECT_NEAR(c.at(0, 0), 22.0f, 1e-4f);
    EXPECT_NEAR(c.at(0, 1), 28.0f, 1e-4f);
    EXPECT_NEAR(c.at(1, 0), 49.0f, 1e-4f);
    EXPECT_NEAR(c.at(1, 1), 64.0f, 1e-4f);
}

TEST(MatrixTest, Transpose) {
    Matrix a(2, 3);
    for (int i = 0; i < 6; ++i) a.data[i] = i;
    auto t = a.transpose();
    ASSERT_EQ(t.rows, 3);
    ASSERT_EQ(t.cols, 2);
    EXPECT_NEAR(t.at(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(2, 1), 5.0f, 1e-6f);
}

TEST(ActivationTest, ReLUForward) {
    Matrix x(1, 4);
    x.data = {-2.0f, -0.5f, 0.0f, 3.0f};
    ReLU relu;
    auto out = relu.forward(x);
    EXPECT_NEAR(out.data[0], 0.0f, 1e-6f);
    EXPECT_NEAR(out.data[1], 0.0f, 1e-6f);
    EXPECT_NEAR(out.data[2], 0.0f, 1e-6f);
    EXPECT_NEAR(out.data[3], 3.0f, 1e-6f);
}

TEST(ActivationTest, SigmoidRange) {
    Matrix x(1, 3);
    x.data = {-10.0f, 0.0f, 10.0f};
    Sigmoid sig;
    auto out = sig.forward(x);
    EXPECT_NEAR(out.data[0], 0.0f, 1e-3f);
    EXPECT_NEAR(out.data[1], 0.5f, 1e-6f);
    EXPECT_NEAR(out.data[2], 1.0f, 1e-3f);
}

TEST(SoftmaxTest, SumsToOne) {
    Matrix x(2, 3);
    x.data = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 2.5f};
    Softmax sm;
    auto out = sm.forward(x);
    float sum0 = out.at(0,0) + out.at(0,1) + out.at(0,2);
    float sum1 = out.at(1,0) + out.at(1,1) + out.at(1,2);
    EXPECT_NEAR(sum0, 1.0f, 1e-5f);
    EXPECT_NEAR(sum1, 1.0f, 1e-5f);
}

TEST(LinearTest, ForwardShape) {
    Linear fc(4, 8);
    Matrix x = Matrix::randn(32, 4);
    auto out = fc.forward(x);
    ASSERT_EQ(out.rows, 32);
    ASSERT_EQ(out.cols, 8);
}

TEST(LinearTest, BackwardGradShape) {
    Linear fc(4, 8);
    Matrix x = Matrix::randn(16, 4);
    fc.forward(x);
    Matrix grad_out = Matrix::randn(16, 8);
    auto grad_in = fc.backward(grad_out);
    ASSERT_EQ(grad_in.rows, 16);
    ASSERT_EQ(grad_in.cols, 4);
}

TEST(CrossEntropyTest, DecreaseWithCorrectPrediction) {
    // If we push logits strongly towards the correct class, loss should be low
    Matrix logits(2, 3, 0.0f);
    logits.at(0, 1) = 10.0f; // class 1 strongly predicted
    logits.at(1, 2) = 10.0f; // class 2 strongly predicted
    std::vector<int> labels = {1, 2};
    CrossEntropyLoss loss;
    float l = loss.forward(logits, labels);
    EXPECT_LT(l, 0.001f);
}

TEST(AdamTest, ParameterUpdates) {
    Linear fc(4, 4);
    Matrix orig_W = fc.W;
    std::vector<ParamRef> params = {{&fc.W, &fc.dW}, {&fc.b, &fc.db}};
    Adam optim(0.01f);
    optim.init(params);
    // Set non-zero gradients
    for (auto& v : fc.dW.data) v = 0.1f;
    for (auto& v : fc.db.data) v = 0.1f;
    optim.step(params);
    bool changed = false;
    for (int i = 0; i < (int)fc.W.data.size(); ++i)
        if (std::abs(fc.W.data[i] - orig_W.data[i]) > 1e-6f) { changed = true; break; }
    EXPECT_TRUE(changed);
}

TEST(DropoutTest, ZeroesInTraining) {
    Dropout drop(0.5f);
    drop.training = true;
    Matrix x(100, 100);
    for (auto& v : x.data) v = 1.0f;
    auto out = drop.forward(x);
    int zeros = 0;
    for (auto v : out.data) if (v == 0.0f) ++zeros;
    // With p=0.5 expect roughly 50% zeros
    EXPECT_GT(zeros, 3000);
    EXPECT_LT(zeros, 7000);
}

TEST(DropoutTest, PassThroughInference) {
    Dropout drop(0.5f);
    drop.training = false;
    Matrix x(10, 10);
    for (auto& v : x.data) v = 1.0f;
    auto out = drop.forward(x);
    for (auto v : out.data) EXPECT_NEAR(v, 1.0f, 1e-6f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
