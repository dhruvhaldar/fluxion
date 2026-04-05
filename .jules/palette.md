## 2026-03-24 - Adding Accessible Focus States and Link Formatting
**Learning:** Native `<a>` tags require clear visual feedback for keyboard users using `:focus-visible` to ensure a11y compliance. Additionally, improving base typography ensures better readability for all users, and opening external links in a new tab requires both security attributes (`rel="noopener noreferrer"`) and an explanatory `aria-label` to warn screen reader users.
**Action:** Always add explicit `:focus-visible` styling (like `outline`) when styling interactive elements natively to provide clear keyboard focus indicators, enhance generic body copy with explicit `line-height`, `color`, and `system-ui` fonts for clarity, and always provide an explanatory `aria-label` when utilizing `target="_blank"`.

## 2026-03-25 - CSS Native Dark Mode with Transitions
**Learning:** Light-sensitive users and those preferring dark environments benefit immensely from native dark mode support. Implementing it using CSS variables (`--var-name`) and the `@media (prefers-color-scheme: dark)` query allows for a smooth, JavaScript-free implementation that is inherently accessible. Adding `transition: background-color 0.3s, color 0.3s;` on the `body` provides a much gentler UX when toggling between themes at the OS level.
**Action:** Always consider `prefers-color-scheme: dark` for web interfaces. Centralizing color definitions in `:root` variables makes it trivial to override them for dark mode, creating a robust, easily maintainable micro-UX improvement.
## 2026-03-26 - Semantic and Descriptive Images for Scientific Visualizations
**Learning:** Raw `<img>` tags in scientific plots often lack context or misrepresent complex data to screen reader users when their `alt` text only repeats the title.
**Action:** When displaying data visualizations or scientific artifacts, wrap them in `<figure>` and `<figcaption>` elements for semantic structure. Furthermore, ensure the `alt` text describes the *content* and *finding* of the plot (e.g., "Log-log plot showing slope of 2.0"), rather than just its title, to ensure robust accessibility.

## 2026-03-31 - Link Accessibility
**Learning:** When link color contrast against surrounding body text falls below the WCAG requirement of 3:1, relying solely on color to distinguish links from text creates an accessibility barrier for users with low vision or color blindness (failing WCAG 1.4.1 Use of Color).
**Action:** Always provide a persistent, non-color visual identifier (such as `text-decoration: underline` or a visible `border-bottom`) for inline links to ensure they are perceivable regardless of color perception.

## 2026-04-01 - Native Dark Mode Scrollbars and Distinct Hover States
**Learning:** Even when implementing a custom CSS dark theme via `@media (prefers-color-scheme: dark)`, browser-native UI elements like scrollbars will remain blindingly bright white unless `color-scheme: light dark;` is explicitly defined on the `:root`. Furthermore, link hover states must be visually distinct from their default states (e.g., changing colors or adding thickness rather than remaining identical) to provide interactive feedback to sighted users.
**Action:** Always include `color-scheme: light dark;` in the `:root` to ensure holistic theme consistency. When designing custom link colors, explicitly verify that the `:hover` color value differs noticeably from the default color value.

## 2026-04-02 - Image Accessibility in Dark Mode
**Learning:** Pure white scientific images can cause eye strain when viewed in dark mode. Furthermore, embedding them directly restricts users from viewing them in full resolution and navigating to them via keyboard.
**Action:** When displaying pure white scientific images or visualizations in dark mode, apply a dimming filter (`filter: brightness(0.85)`) to reduce eye strain. Always wrap these images in focusable anchor tags for full-resolution access, and use `:hover`/`:focus-visible` states to restore full brightness (`filter: brightness(1)`) for detailed inspection and accessibility parity.
