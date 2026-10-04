# Results

PSNR/SSIM compare each rendering with the photo, so they reward copying pixels, not simplifying. Read them next to the path count and file size.

## canal

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 14985 | 16433 | 23.0 | 0.767 |
| vtracer (matched paths) | 126 | 1875 | 18.9 | 0.508 |
| vtracer 1.0 (watershed, matched paths) | 125 | 206 | 17.2 | 0.411 |
| vtracer 1.0 (colour, matched paths) | 125 | 736 | 18.1 | 0.482 |
| SuperSVG (CVPR 2024) | 126 | 69 | 18.1 | 0.412 |
| tracesmart (automatic) | 122 | 58 | 15.6 | 0.369 |
| tracesmart (--care) | 126 | 56 | 15.8 | 0.368 |

## delft-street

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 12652 | 12215 | 24.1 | 0.765 |
| vtracer (matched paths) | 235 | 280 | 10.9 | 0.356 |
| vtracer 1.0 (watershed, matched paths) | 232 | 176 | 18.3 | 0.429 |
| vtracer 1.0 (colour, matched paths) | 236 | 461 | 18.6 | 0.498 |
| SuperSVG (CVPR 2024) | 234 | 128 | 18.4 | 0.406 |
| tracesmart (automatic) | 157 | 73 | 15.3 | 0.374 |
| tracesmart (--care) | 234 | 80 | 17.1 | 0.38 |

## hikers

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 17699 | 19404 | 24.5 | 0.781 |
| vtracer (matched paths) | 128 | 1215 | 12.4 | 0.392 |
| vtracer 1.0 (watershed, matched paths) | 128 | 212 | 17.2 | 0.383 |
| vtracer 1.0 (colour, matched paths) | 128 | 744 | 18.4 | 0.478 |
| SuperSVG (CVPR 2024) | 127 | 69 | 18.7 | 0.385 |
| tracesmart (automatic) | 139 | 89 | 15.5 | 0.359 |
| tracesmart (--care) | 127 | 75 | 15.6 | 0.358 |

## lone-hiker

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 11742 | 13745 | 19.6 | 0.73 |
| vtracer (matched paths) | 86 | 1885 | 17.6 | 0.533 |
| vtracer 1.0 (watershed, matched paths) | 83 | 168 | 19.1 | 0.473 |
| vtracer 1.0 (colour, matched paths) | 86 | 577 | 17.6 | 0.523 |
| SuperSVG (CVPR 2024) | 83 | 45 | 19.0 | 0.459 |
| tracesmart (automatic) | 87 | 53 | 16.5 | 0.452 |
| tracesmart (--care) | 83 | 51 | 17.3 | 0.442 |

## mountain-lake

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 9127 | 10415 | 22.6 | 0.751 |
| vtracer (matched paths) | 104 | 999 | 13.8 | 0.577 |
| vtracer 1.0 (watershed, matched paths) | 105 | 190 | 19.9 | 0.582 |
| vtracer 1.0 (colour, matched paths) | 102 | 505 | 20.3 | 0.618 |
| SuperSVG (CVPR 2024) | 105 | 58 | 22.0 | 0.576 |
| tracesmart (automatic) | 64 | 41 | 18.4 | 0.569 |
| tracesmart (--care) | 105 | 55 | 19.6 | 0.571 |

## oostpoort

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 18237 | 20325 | 23.3 | 0.75 |
| vtracer (matched paths) | 96 | 2637 | 18.4 | 0.468 |
| vtracer 1.0 (watershed, matched paths) | 95 | 177 | 16.8 | 0.384 |
| vtracer 1.0 (colour, matched paths) | 95 | 736 | 18.2 | 0.46 |
| SuperSVG (CVPR 2024) | 95 | 52 | 18.1 | 0.386 |
| tracesmart (automatic) | 71 | 55 | 16.1 | 0.366 |
| tracesmart (--care) | 95 | 65 | 16.6 | 0.369 |

## Average over 6 photos

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 14074 | 15423 | 22.8 | 0.76 |
| vtracer (matched paths) | 129 | 1482 | 15.3 | 0.47 |
| vtracer 1.0 (watershed, matched paths) | 128 | 188 | 18.1 | 0.44 |
| vtracer 1.0 (colour, matched paths) | 129 | 626 | 18.5 | 0.51 |
| SuperSVG (CVPR 2024) | 128 | 70 | 19.1 | 0.44 |
| tracesmart (automatic) | 107 | 62 | 16.2 | 0.41 |
| tracesmart (--care) | 128 | 64 | 17.0 | 0.41 |
