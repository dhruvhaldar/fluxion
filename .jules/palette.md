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

## 2026-04-03 - Interactive Linked Images
**Learning:** While filtering image brightness on hover provides a good dark-mode interaction, users in light-mode (or default) environments may not realize embedded scientific visualization images are clickable links to view them in full resolution.
**Action:** To ensure clear interactivity for linked images (such as scientific visualizations) across all color schemes, apply universal visual feedback for `:hover` and `:focus-visible` states directly to the image element (e.g., modifying `border-color` and `box-shadow` on `a:hover img` and `a:focus-visible img`).

## 2026-04-04 - Image Link Cursors and Focus Wrappers
**Learning:** When images are wrapped in anchor tags to allow users to view full-resolution versions, the default `cursor: pointer` is less descriptive than `cursor: zoom-in`, which explicitly communicates the action. Furthermore, because anchor tags are `display: inline` by default, keyboard focus rings (`:focus-visible`) on anchors wrapping block-level images often render awkwardly (e.g., zero height or misaligned).
**Action:** Always apply `cursor: zoom-in` to links that expand images. Ensure image-wrapping anchor tags are explicitly set to `display: block` (or `inline-block`) with a matching `border-radius` so that native keyboard focus rings properly encompass the entire image.

## 2026-04-08 - Accessible External Links and Visible Text
**Learning:** Using `aria-label` on links that contain visible text is an anti-pattern. An `aria-label` entirely replaces the visible text for screen reader users and can break voice dictation software (like Dragon) because the accessible name no longer matches the visible text on the screen.
**Action:** When adding supplemental information to links with visible text (such as "opens in a new tab" for `target="_blank"`), avoid `aria-label`. Instead, use a visually hidden `<span>` (e.g., using a `.sr-only` utility class) immediately following the visible text inside the anchor tag.


## 2026-04-09 - Native Tooltips for Image Links
**Learning:** For interfaces displaying scientific visualization images wrapped in `<a>` tags with `target="_blank"`, relying on `cursor: zoom-in` CSS alone isn't sufficient for all users to understand the interaction. Adding a native HTML `title` attribute directly to the `<a>` element provides a reliable, accessible tooltip describing the interaction ("Click to view full size") without requiring any custom CSS or complex JS tooltip libraries.
**Action:** Always add native `title` attributes to links wrapping full-resolution images to explicitly communicate the expected interaction to the user, especially when constrained from using custom CSS.

## 2026-04-12 - Semantic Contentinfo Landmarks and Fluid Icons
**Learning:** Burying supplementary content (like external repository links) inside the main document flow reduces navigational clarity for screen readers. Using a `<footer>` tag establishes a semantic `contentinfo` landmark, allowing users to easily jump to the end of the document. Additionally, hardcoded pixel sizes on inline icons break layout when users adjust base font sizes.
**Action:** Always extract trailing document metadata or global links into a `<footer>` to define a `contentinfo` landmark. When including inline SVG icons alongside text, use `width="1em" height="1em"` instead of pixel values so the icon fluidly scales with user font preferences.

## 2026-04-13 - Anchor Links and Scroll Padding
**Learning:** When using internal anchor links (like "Skip to main content"), the browser scrolls the target element directly to the top edge of the viewport. This often bypasses the document's `body` padding, causing content to flush uncomfortably against the window edge.
**Action:** Always add `scroll-padding-top` (matching the `body` padding, e.g., `20px`) to the `html` element. This preserves visual spacing and layout integrity when keyboard and screen-reader users utilize in-page navigation links.

## 2026-04-14 - Back to Top Links on Long Artifact Pages
**Learning:** On pages with multiple large scientific figures (artifacts), scrolling back to the top navigation or introductory content can be tedious for users. Relying purely on scrollbars or keyboard page-up is less intuitive than a clear interaction point.
**Action:** Always provide a "Back to Top" link at the bottom of long, image-heavy pages. Implementing it natively with an internal anchor link (e.g., `href="#top"`) and pairing it with CSS `scroll-behavior: smooth` provides an accessible, performant, and delightful micro-UX improvement.

## 2026-04-15 - Permalink Heading Anchors
**Learning:** Document-heavy pages with multiple sections (like scientific reports or long artifact lists) can be difficult for users to share specific references to. Adding anchor tags (`#`) deep-linking directly to sections significantly improves the utility and sharing UX. However, having them permanently visible creates visual noise.
**Action:** Append a permalink anchor inside headings and style them to be visually hidden (`opacity: 0`) by default, revealing them only on `:hover` or `:focus-visible`. This provides the utility of deep linking without compromising visual cleanliness, while remaining fully accessible to keyboard users.

## 2025-02-12 - Internal Anchor Link Focus Management
**Learning:** When adding permalink anchor tags (like "#") directly inside headings (`<h2>`) to enable deep-linking to sections, screen readers require the heading element to programmatically manage focus correctly. Without `tabindex="-1"`, deep-linking only scrolls the page visually but does not relocate screen reader focus, leaving users disoriented.
**Action:** Always add `tabindex="-1"` to target heading elements when implementing deep-link anchors inside them. Also, pair it with `outline: none` on the `:focus` pseudo-class (e.g. `h2:focus { outline: none; }`) to suppress the default visual focus ring for sighted users while preserving correct programmatic focus.

## 2026-04-21 - Body Focus and Print Accessibility
**Learning:** Adding a "Back to top" link pointing to `#top` on the `<body>` element provides a great micro-interaction, but without `tabindex="-1"` on the `<body>`, screen reader and keyboard focus doesn't properly reset. Furthermore, scientific artifacts in web pages are frequently printed or saved as PDFs. Dark mode styles and navigational elements (like "Skip to content" or anchor links) clutter the print output.
**Action:** Always add `tabindex="-1"` to the target element (even the `<body>`) when implementing internal jump links to ensure programmatic focus resets, and pair it with `:focus { outline: none; }` to hide the massive focus ring. Additionally, implement a `@media print` stylesheet to hide navigational UI elements and force high-contrast print colors for artifact readability.

## 2026-04-22 - Print Accessibility for External Links
**Learning:** In document-heavy interfaces (such as scientific reports) that are frequently printed or exported to PDF, visually hidden elements meant for screen readers (like `.sr-only` text) and decorative icons (like SVG external link indicators) create messy artifacts on physical pages. More critically, external hyperlinks lose all utility when printed if the destination URL isn't visible.
**Action:** Always enhance `@media print` stylesheets to explicitly hide `.sr-only` utility classes and decorative icons. Additionally, use CSS generated content (`a[href^="http"]::after { content: " (" attr(href) ")"; }`) to print the actual destination URLs next to external links, ensuring the document remains functional offline.

## 2026-04-24 - Visual Feedback for Deep Links
**Learning:** When deep-linking to internal page sections, users can become disoriented. Providing temporary visual feedback via the `:target` pseudo-class improves navigational orientation.
**Action:** Always provide visual highlighting (like a temporary color pulse) using the `:target` pseudo-class when users jump to in-page anchors.

## 2026-04-25 - Touch Device Accessibility for Hover-Revealed Elements
**Learning:** Hiding utility elements (like permalink anchors) by default with `opacity: 0` and revealing them via `:hover` and `:focus-visible` creates a clean UI for mouse and keyboard users. However, touch device users (like mobile and tablet users) lack hover capability. If they don't use external keyboards, these elements become completely undiscoverable and permanently inaccessible, breaking core navigational utility for a large segment of users.
**Action:** When hiding interactive elements by default to reduce visual clutter, always include a `@media (hover: none)` block to ensure the elements remain permanently visible (e.g., `opacity: 1` or a semi-transparent state) for touch device users who cannot trigger hover states.

## 2026-04-28 - Redundant Alt Text in Figures
**Learning:** When images are wrapped in `<figure>` tags and paired with a descriptive `<figcaption>`, replicating the caption text inside the image's `alt` attribute creates a frustrating, redundant double-reading experience for screen reader users. Furthermore, repeating the title of a scientific plot in the `alt` text fails to convey the actual visual information (the trend or shape of the data) that sighted users receive.
**Action:** When an image has an adjacent `<figcaption>`, ensure the `alt` text describes the *visual content* of the image (e.g., 'A line graph showing an upward curve') to provide equivalent informational value without repeating the caption.

## 2026-05-01 - Context-Specific Permalink Labels
**Learning:** Using generic labels like `aria-label="Permalink to this section"` on heading anchor links creates a frustrating experience for screen reader users who navigate by links, as they hear the same uninformative phrase repeated.
**Action:** Always provide context-specific `aria-label` and `title` attributes on internal section anchors (e.g., `Permalink to [Heading Name]`) to improve navigability and context.

## 2026-05-02 - Linking Image Wrapper Links to Captions
**Learning:** When images are wrapped in `<figure>` tags and `<a>` tags, adding an `aria-describedby` attribute to the link pointing to the `<figcaption>` element's `id` ensures that screen reader users navigating via links hear both the image's `alt` text and the surrounding caption context.
**Action:** When wrapping images inside `<figure>` elements with links, always link the `<a>` tag to the `<figcaption>` using `aria-describedby` and matching `id` attributes.

## 2026-05-03 - Keyboard Interaction Parity for Micro-Interactions
**Learning:** When micro-interactions (such as link color shifts, underline thickness changes, or SVG icon translations) are mapped exclusively to the `:hover` pseudo-class, keyboard users navigating via Tab do not receive the same visual feedback or polish, experiencing only the default or custom focus ring. This breaks interaction parity and visual delight for keyboard-only users.
**Action:** Always pair `:focus-visible` with `:hover` selectors for delightful state changes (e.g., `a:hover, a:focus-visible`) to ensure keyboard users experience the exact same visual micro-interactions as mouse users.

## 2026-05-07 - Typographic Polish and Print Layout Flow
**Learning:** For document-heavy interfaces meant for reading and printing, CSS text-wrapping properties significantly improve readability by preventing typographic widows. Furthermore, when physical printing is common, preventing awkward page breaks (like splitting a figure from its caption or orphaning a heading at the bottom of a page) is a critical UX consideration.
**Action:** Use `text-wrap: balance` for headings and `text-wrap: pretty` for paragraphs/captions. Always include `break-inside: avoid` on `<figure>` elements and `break-after: avoid` on headings within `@media print` stylesheets to ensure logical document flow on physical pages.
