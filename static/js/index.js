document.addEventListener('DOMContentLoaded', function () {
  // Chart.js setup
  const data = [
    { label: 'MobileNeRF', fps: 24, psnr: 29.3, memory: 194, color: '#7f7f7f' },
    { label: '3DGS-50K', fps: 20, psnr: 32.73, memory: 12, color: '#1f77b4' },
    { label: '3DGS-75K', fps: 13, psnr: 33.05, memory: 18, color: '#ff7f0e' },
    { label: '3DGS', fps: 8, psnr: 35.44, memory: 57, color: '#2ca02c' },
    { label: '3-Mesh', fps: 65, psnr: 33.39, memory: 46, color: '#d62728' },
    { label: '5-Mesh', fps: 55, psnr: 34.25, memory: 77, color: '#9467bd' },
    { label: '7-Mesh', fps: 42, psnr: 34.50, memory: 110, color: '#8c564b' },
    { label: '9-Mesh', fps: 35, psnr: 34.38, memory: 140, color: '#e377c2' }
  ];

  const dataset = {
    datasets: data.map(point => ({
      label: point.label,
      data: [{ x: point.fps, y: point.psnr, r: Math.sqrt(point.memory) }],
      backgroundColor: point.color,
      borderColor: '#333',
      borderWidth: 1
    }))
  };

  // Register the plugins
  Chart.register(ChartDataLabels);
  Chart.register(ChartDataLabels, window['chartjs-plugin-annotation']);

  // Chart config with options and plugin
  const config = {
    type: 'bubble',
    data: dataset,
    options: {
      plugins: {
        datalabels: {
          align: 'top',
          anchor: 'end',
          font: {
            size: 10,
            weight: 'bold'
          },
          formatter: (value, context) => context.dataset.label,
          color: '#000'
        },
        tooltip: {
          callbacks: {
            label: ctx => {
              const pt = ctx.raw;
              const label = ctx.dataset.label;
              const mem = (pt.r ** 2).toFixed(1);
              return `${label}: ${pt.x} FPS, ${pt.y} PSNR, ${mem} MB`;
            }
          }
        },
        legend: { display: false },
        annotation: {
          annotations: {
            xLine: {
              type: 'line',
              xMin: 30,
              xMax: 30,
              borderColor: 'black',
              borderWidth: 1,
              borderDash: [4, 4],
              label: {
                display: true,
                content: 'real-time',
                position: 'start',
                color: 'black',
                backgroundColor: 'transparent',
                font: {
                  size: 10,
                  style: 'italic'
                }
              }
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'FPS ⋄' },
          min: 0,
          max: 70
        },
        y: {
          title: { display: true, text: 'PSNR' },
          min: 28,
          max: 36
        }
      }
    }
  };

  // Create chart
  const ctx = document.getElementById('performanceChart');
  if (ctx) new Chart(ctx, config);
});