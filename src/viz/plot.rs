use plotters::prelude::*;

use crate::model::linear_regression::{estimate_price, Model};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const POINT_RADIUS: i32 = 4;

pub type PlotResult = Result<(), Box<dyn std::error::Error>>;

fn axis_bounds(values: &[f64]) -> (f64, f64) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = (max - min) * 0.05;
    (min - pad, max + pad)
}

/// 학습 데이터의 (km, price) 산점도만 그려서 저장.
pub fn scatter(xs: &[f64], ys: &[f64], path: &str) -> PlotResult {
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_lo, x_hi) = axis_bounds(xs);
    let (y_lo, y_hi) = axis_bounds(ys);

    let mut chart = ChartBuilder::on(&root)
        .caption("Mileage vs Price (training data)", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("km")
        .y_desc("price")
        .draw()?;

    chart.draw_series(
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| Circle::new((x, y), POINT_RADIUS, BLUE.filled())),
    )?;

    root.present()?;
    Ok(())
}

/// 산점도 위에 회귀선을 겹쳐 그리고, 옵션으로 예측 지점(km, ŷ)을 강조 표시.
pub fn regression(
    xs: &[f64],
    ys: &[f64],
    model: &Model,
    highlight: Option<(f64, f64)>,
    path: &str,
) -> PlotResult {
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    // 강조 지점이 데이터 범위를 벗어날 수 있으니 함께 고려해 축 범위를 잡는다.
    let mut xs_for_axis = xs.to_vec();
    let mut ys_for_axis = ys.to_vec();
    if let Some((hx, hy)) = highlight {
        xs_for_axis.push(hx);
        ys_for_axis.push(hy);
    }
    let (x_lo, x_hi) = axis_bounds(&xs_for_axis);
    let (y_lo, y_hi) = axis_bounds(&ys_for_axis);

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear regression", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("km")
        .y_desc("price")
        .draw()?;

    chart
        .draw_series(
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| Circle::new((x, y), POINT_RADIUS, BLUE.filled())),
        )?
        .label("data")
        .legend(|(x, y)| Circle::new((x + 10, y), POINT_RADIUS, BLUE.filled()));

    let line = vec![
        (x_lo, estimate_price(model, x_lo)),
        (x_hi, estimate_price(model, x_hi)),
    ];
    chart
        .draw_series(LineSeries::new(line, RED.stroke_width(2)))?
        .label("regression line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    if let Some((hx, hy)) = highlight {
        chart
            .draw_series(std::iter::once(Cross::new((hx, hy), 10, GREEN.stroke_width(3))))?
            .label(format!("prediction ({hx:.0} km → {hy:.0})"))
            .legend(|(x, y)| Cross::new((x + 10, y), 6, GREEN.stroke_width(3)));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
