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

## 2026-05-10 - Preventing Double Focus Rings on Linked Images
**Learning:** When adding custom `:focus-visible` styles to images wrapped inside anchor tags within a `<figure>` element (e.g., to improve visual feedback for scientific visualizations), the browser also applies a default focus ring to the wrapping `<a>` element. This results in a confusing and messy double-focus ring when keyboard users navigate to the image.
**Action:** When applying custom focus styles to `img` elements inside linked figures (`a:focus-visible img`), always explicitly remove the default focus ring on the parent anchor tag by adding `figure a:focus-visible { outline: none; }` to maintain a clean interaction state.
## 2026-05-12 - Explicit Interactive Affordance for <abbr> tags
**Learning:** Relying entirely on browser default styling for `<abbr>` tags leaves them undiscoverable since they look identical to regular text, meaning users miss out on helpful tooltips.
**Action:** When using `<abbr title="...">`, always add `text-decoration: underline dotted` and `cursor: help` to ensure the element clearly signals interactivity.

## 2026-05-14 - Print Accessibility for Abbreviations
**Learning:** In document-heavy interfaces that are frequently printed or exported to PDF, `<abbr>` tags with `title` attributes lose their utility because the expanded definition isn't visible.
**Action:** Always enhance `@media print` stylesheets to append the `title` attribute of an `<abbr>` tag using CSS generated content (e.g., `abbr[title]::after { content: " (" attr(title) ")"; }`) to ensure acronym definitions are visible on physical prints.

## 2026-05-15 - Keyboard Focus for <abbr> Tooltips
**Learning:** By default, HTML `<abbr>` elements with `title` attributes only show tooltips on hover, rendering the expanded definition inaccessible to sighted keyboard users who cannot hover.
**Action:** When using `<abbr>` elements for tooltips, always add `tabindex="0"` to make them keyboard-focusable, and ensure they have a `:focus-visible` outline in CSS for clear interactive feedback.

## 2026-05-17 - Keyboard Accessible Abbreviations Tooltips
**Learning:** When using `<abbr>` elements for tooltips, relying solely on the default `title` attribute renders the expanded definition inaccessible to sighted keyboard users who cannot hover. To achieve visible parity for keyboard focus, a custom CSS tooltip pseudo-element (e.g., `::before` with `content: attr(title)`) triggered by `:focus-visible` must be implemented. However, care must be taken to explicitly hide this pseudo-element in `@media print` to avoid generating duplicate text or visual artifacts alongside the printed definitions (which typically use `::after`).
**Action:** When implementing keyboard-accessible tooltips on `<abbr>` elements, always add `tabindex="0"` to make them focusable, and use a `:focus-visible` triggered pseudo-element to display the `title` attribute. Ensure the tooltip is hidden in print stylesheets (`@media print`) and use `::before` to avoid conflicts if `::after` is used to append definitions for physical prints.

## 2026-05-18 - Keyboard Accessible Tooltips for Links
**Learning:** Browsers only display the native `title` attribute tooltip on hover. When adding a `title` attribute to links (such as `<a>` tags wrapping images) to provide helpful context (e.g., "Click to view full size"), sighted keyboard users are entirely excluded from this information because they cannot trigger the hover state.
**Action:** When relying on the `title` attribute on links for important contextual information or utility, provide interaction parity by creating a custom CSS tooltip (e.g., using `::after` with `content: attr(title)`) triggered on the `:focus-visible` state. Ensure these custom tooltips are explicitly hidden in `@media print` stylesheets to prevent printing artifacts.

## 2026-05-19 - Touch Accessibility for Custom Tooltips
**Learning:** While mapping custom CSS tooltips (such as for `<abbr>` elements) to `:hover` and `:focus-visible` covers mouse and keyboard users, it entirely excludes touch device users. Touch devices (like mobile/tablets) don't trigger hover easily, and tapping an element often does not reliably trigger `:focus-visible` to display the tooltip, rendering definitions undiscoverable.
**Action:** When implementing custom CSS tooltips, always include a `@media (hover: none)` block mapping the tooltip visibility to the `:active` state. This ensures that users on touch devices can view the tooltip while pressing down on the element. Ensure to explicitly hide this `:active` state tooltip in `@media print` to prevent print artifacts.

## 2026-05-20 - Touch Accessibility for Custom Link Tooltips
**Learning:** Adding custom CSS tooltips (via `::after` triggered on `:focus-visible`) to image links allows keyboard users to access `title` attribute context, but completely excludes touch device users who cannot trigger hover or robust focus states.
**Action:** When creating custom CSS tooltips for `title` attributes on anchor tags or other elements, always include a `@media (hover: none)` block mapping the tooltip visibility to the `:active` pseudo-class (e.g., `a[title]:active::after`). This enables touch device users to view the tooltip while long-pressing the element. Ensure to explicitly hide this `:active` state tooltip in `@media print` to prevent print artifacts.

## 2026-05-21 - Responsive Custom Tooltips
**Learning:** Using `white-space: nowrap` on custom CSS tooltips (such as those generated for `abbr[title]`) causes the tooltip box to overflow the viewport horizontally on narrow mobile screens if the text is too long, rendering parts of it unreadable and creating awkward scrollbars.
**Action:** Always make custom tooltips wrap gracefully by replacing `white-space: nowrap` with `white-space: normal; width: max-content; max-width: min(250px, 90vw); text-align: center;`. This ensures the tooltip scales nicely while preventing text from running off the screen.

## 2026-05-22 - Skip Link Visibility with Font Scaling
**Learning:** Hiding "skip to content" links using a fixed negative offset (e.g., `top: -40px`) breaks when users increase their browser's default font size or zoom. As the text scales up, the height of the link increases beyond the fixed 40px, causing the bottom of the link to peek out permanently at the top of the viewport.
**Action:** Always hide skip links dynamically based on their own computed height by using `transform: translateY(-100%)`. On focus, reveal the link with `transform: translateY(0)`. This ensures the link remains completely hidden by default, regardless of how much the user scales the font.
## 2026-05-23 - Interactive Image Focus and Lift States
**Learning:** Simply changing the border color or shadow of an image inside an anchor tag is often insufficient as a focus indicator for sighted keyboard users. Replacing a native focus ring requires adding a clear, visible `outline` specifically mapped to the `:focus-visible` state of the linked image. Furthermore, augmenting large `box-shadow` changes with a `transform: translateY` provides a more intuitive, physical sense of the element "lifting" off the page for interactive states.
**Action:** When overriding default focus rings on linked images, ensure an explicit `outline` with adequate offset is supplied on the `:focus-visible` state, and pair heavy shadow transitions with a slight `translateY` offset to match visual expectations.

## 2026-05-24 - Custom Text Selection Highlight
**Learning:** By default, browsers apply a generic blue or grey highlight color during text selection (via `::selection`). On pages with carefully crafted color palettes or distinct dark mode implementations, this default styling often clashes with the design system and can result in suboptimal text contrast.
**Action:** Always define custom `::selection` (and `::-moz-selection`) styles using semantic CSS variables (e.g., `background-color: var(--link-color); color: var(--bg-color);`) to ensure text selection remains highly legible, brand-aligned, and automatically responsive to light/dark mode transitions.

## 2026-05-25 - Tactile Active States for Lift Interactions
**Learning:** When using CSS transforms to "lift" elements (e.g., `translateY(-2px)`) on `:hover` or `:focus-visible`, failing to add a corresponding `:active` state that pushes the element back down to its original position (e.g., `translateY(0)`) makes the interaction feel unresponsive and floaty.
**Action:** Always pair lift animations with an `:active` state that counteracts the transform and reduces the `box-shadow` to provide clear, tactile confirmation of a click or tap.

## 2026-05-30 - Persistent Interactive Affordance for Touch Devices
**Learning:** Relying purely on `cursor` or hover/focus-revealed states for linked images (like zoom interactions) leaves touch device users without any visual affordance that the element is interactive, breaking feature discoverability.
**Action:** Always provide a persistently visible interactive affordance (like a subtle, overlaid icon) on interactive image links to ensure touch-device users know they can interact with the element. Enhance this affordance with `:hover` and `:focus-visible` states for visual delight, and explicitly hide it in `@media print` stylesheets.
## 2024-05-24 - Heading anchor link tooltips for keyboard users
**Learning:** Native `title` attributes on structural anchor links (like heading permalinks e.g. `#`) are inaccessible to keyboard users because they only display on hover, leaving focus users with only an ambiguous `#` symbol.
**Action:** When using `title` attributes on minimalist structural links, provide interaction parity by explicitly surfacing the title text via custom CSS pseudo-element tooltips (e.g. `::after` with `content: attr(title)`) triggered by the `:focus-visible` state.

## 2026-06-04 - Heading Permalinks Text Selection
**Learning:** Heading permalinks (often represented by a `#` symbol next to the heading text) are a common UX pattern. However, if a user double-clicks the heading text to quickly select and copy it, the permalink symbol `#` is often unintentionally selected as well. This leads to frustrating cleanup when pasting the copied heading.
**Action:** Always add `user-select: none;` to the CSS class for heading permalinks (e.g., `.heading-anchor`). This simple addition ensures the anchor symbol is ignored during text selection, allowing users to cleanly copy the heading text without accidentally grabbing the `#` character.

## 2026-06-05 - Native Tap Highlights vs Custom Active States
**Learning:** By default, mobile browsers (like iOS Safari and Android Chrome) apply a grey semi-transparent overlay to interactive elements when tapped. When implementing custom `:active` animations (such as a "lift and press" effect or custom pseudo-element tooltips mapped to `:active`), this native tap highlight visibly clashes with the custom interaction, resulting in a muddy, unresponsive-feeling experience.
**Action:** When implementing custom `:active` interactions on mobile/touch elements (like `<a>` or `<abbr>`), always suppress the default browser tap highlight by adding `-webkit-tap-highlight-color: transparent;` to the element. Furthermore, ensure tiny interactive elements like heading permalinks (`#`) have an expanded touch target (e.g., using a negatively positioned `::before` pseudo-element) to prevent frustrating tap-misses on mobile screens.

## 2026-06-13 - "Back to Top" Link Accessibility
**Learning:** Native `<a href="#top">` tags containing both an icon symbol and text might not provide sufficient context for screen reader users when read out of context.
**Action:** When a "Back to top" link is present, especially one with symbolic icons (like arrows), it's best practice to add a clear, descriptive `aria-label` (e.g., `aria-label="Back to top of page"`) to explicitly describe its function, ensuring an accessible experience.
## 2026-06-25 - Absolute Positioning of Skip Links
**Learning:** When hiding 'skip to content' links using absolute positioning and `transform: translateY(-100%)`, failing to explicitly define `top: 0;` causes the element to align to its natural document flow position. If the parent container (like `body`) has padding or margins, the skip link may partially peek into the viewport by default instead of being fully hidden off-screen.
**Action:** Always explicitly set `top: 0;` when using absolute positioning and negative `translateY` transforms to reliably hide skip links at the top edge of the viewport.
## $(date +%Y-%m-%d) - Prevent iOS Context Menu Interference on Custom Touch Tooltips
**Learning:** When implementing custom CSS tooltips mapped to the `:active` state for touch devices, the native iOS context menu (triggered by long presses on interactive elements like links) can appear and visually overlap or block the intended custom tooltip, confusing the user.
**Action:** Always apply `-webkit-touch-callout: none;` to interactive elements (like `<a>` or `<abbr>`) that utilize custom `:active` state tooltips or interactions to suppress the native iOS context menu and ensure a smooth, app-like experience.

## $(date +%Y-%m-%d) - Device-Agnostic Context Terminology
**Learning:** Hardcoding interaction terminology like "Click to view" implicitly assumes the user is operating a mouse or trackpad, which creates cognitive dissonance and excludes users relying on touchscreens (who "tap") or keyboards (who press "Enter" or "Space").
**Action:** Always use device-agnostic terminology (e.g., "View full size" or "Select to expand") for titles, ARIA labels, and helper text to ensure the instructions are inclusive and accurate regardless of the input device.

## $(date +%Y-%m-%d) - CSS Tooltip Viewport Clipping on Anchor Links
**Learning:** When using custom CSS tooltips that display directly above an element (e.g., `bottom: 100%`), navigating to the element via an anchor link (like `#heading`) often scrolls the element flush with the top of the viewport. Consequently, the tooltip renders entirely off-screen, rendering it invisible to sighted keyboard users who trigger the focus state.
**Action:** When implementing top-positioned custom tooltips on elements that act as anchor targets, ensure `scroll-padding-top` on the `html` or `body` element is sufficiently large (e.g., `64px`) to leave enough breathing room above the focused element for the tooltip to fully display without being clipped by the viewport edge.

## 2026-07-18 - Descriptive ARIA Labels for Generic External Links
**Learning:** Using generic link text like "GitHub Repository" alongside an external link icon lacks specific context for screen reader users when read out of context.
**Action:** When creating external links to project repositories, always provide a descriptive `aria-label` (e.g., `aria-label="View the Fluxion CFD GitHub Repository"`) to ensure clear and inclusive navigation.

## $(date +%Y-%m-%d) - Unified SVG Iconography over Unicode
**Learning:** Using native unicode characters (like `↑` or `►`) for UI icons introduces significant cross-OS and cross-browser rendering inconsistencies. These characters often render with different glyph styles, bounding boxes, or even as colorful emojis on some platforms (e.g., iOS), which breaks design cohesion, alignment, and stroke-weight matching with existing SVG icons.
**Action:** Always replace standalone unicode UI icons with explicit, inline SVG icons (e.g., using Feather or a unified icon library) to guarantee pixel-perfect rendering consistency, precise alignment, and cohesive stroke-weight styling across all devices and browsers.
## 2026-07-31 - [Heading Permalink SVG Icon Replacement]
**Learning:** Standalone text characters (like `#`) used as UI icons (e.g., for heading permalinks) lack professional visual polish. When replacing them with explicit SVG icons for better consistency, if custom CSS classes are not permitted and existing utilities are missing, it is necessary to use inline styles for minor alignments (like `vertical-align`). Additionally, these decorative SVGs must include `aria-hidden="true"` so that screen readers continue to rely on the parent anchor tag’s `aria-label`.
**Action:** Replace text-based UI icons with inline SVGs featuring `aria-hidden="true"`, and use inline styling for basic alignment if CSS class additions are restricted.

## 2026-08-01 - Preventing Accidental Semantic Animations from Utility Classes
**Learning:** Reusing utility classes (like `external-link-icon`) strictly for minor layout or alignment purposes on UI components can inadvertently attach unwanted interaction semantics (such as hover/active transform animations intended only for external links) to internal navigation elements, causing a disjointed UX.
**Action:** Ensure that utility classes used to position UI icons do not carry unintended interaction states. If restricted from creating new utility classes to fix the issue, fall back to inline styles (e.g., `style="vertical-align: middle;"`) to correct the interaction semantic mismatch while adhering to design system constraints.
## 2026-08-04 - CSS Transform Overrides on Centered Elements
**Learning:** When using `transform: translate(-50%, -50%)` to center elements (like a pseudo-element zoom icon), any interaction states (`:hover`, `:active`) that apply scale or other transforms will overwrite the translate function if not explicitly included in the transform stack, causing the element to abruptly jump out of center.
**Action:** Always ensure the full transform stack (including the translation used for positioning) is preserved when overriding transformations for interactive states.
## 2026-08-05 - Accessible Fallback Text for Image Anchor Links
**Learning:** When an `<a>` tag wraps an `<img>` that contains important, descriptive `alt` text, adding an `aria-label` directly to the `<a>` tag to describe the interaction (e.g., 'View full size') is an anti-pattern. This is because the `aria-label` on the parent completely overwrites its child content in the accessibility tree, thereby hiding the image's critical descriptive `alt` text from screen readers.
**Action:** When adding descriptive names to image anchor links, instead of using `aria-label`, inject a visually hidden helper `<span>` (e.g., `<span class="sr-only">View full size image</span>`) alongside the image. This preserves both the action description and the image's native `alt` text for assistive technologies.
