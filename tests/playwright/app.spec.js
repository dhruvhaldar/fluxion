const { test, expect } = require('@playwright/test');

test.describe('Fluxion CFD web interface', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the local flask server before each test
    await page.goto('/');
  });

  test('should display the correct page title', async ({ page }) => {
    // Check if title contains "Fluxion CFD"
    await expect(page).toHaveTitle(/Fluxion CFD/);
  });

  test('should display the main heading', async ({ page }) => {
    // Verify the presence of the <h1>Fluxion</h1> heading
    const heading = page.locator('h1');
    await expect(heading).toHaveText('Fluxion');
  });

  test('should display all artifact images', async ({ page }) => {
    // Verify the first image: Lid Driven Cavity Streamlines
    const img1 = page.locator('img[src="/assets/lid_driven_streamlines.png"]');
    await expect(img1).toBeVisible();

    // Verify the second image: Grid Convergence Study
    const img2 = page.locator('img[src="/assets/grid_convergence.png"]');
    await expect(img2).toBeVisible();

    // Verify the third image: Convection Scheme Comparison
    const img3 = page.locator('img[src="/assets/scheme_comparison.png"]');
    await expect(img3).toBeVisible();
  });

  test('should load artifact images properly without breaking', async ({ page }) => {
    // Get all image elements
    const images = page.locator('img');
    const count = await images.count();

    // We expect 3 images on the page
    expect(count).toBe(3);

    // Verify that the images load correctly (checking for 200 OK via network response)
    // Here we can actually verify that the src URLs return a valid response
    const srcPromises = [];
    for (let i = 0; i < count; i++) {
        const src = await images.nth(i).getAttribute('src');
        srcPromises.push(src);
    }

    for (const src of srcPromises) {
        const response = await page.request.get(src);
        expect(response.ok()).toBeTruthy();
    }
  });

  test('should have a link to the GitHub repository', async ({ page }) => {
    const link = page.locator('a', { hasText: 'GitHub Repository' });
    await expect(link).toBeVisible();
    await expect(link).toHaveAttribute('href', 'https://github.com/dhruvhaldar/fluxion');
  });

  test('should capture a screenshot of the main page', async ({ page }) => {
    // Take a screenshot of the whole page
    await page.screenshot({ path: 'assets/playwright_screenshot.png', fullPage: true });
  });
});
