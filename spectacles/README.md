# Spectacles drive the arms

Each hand moves the arm on its side; thumb and index apart open its jaw,
together close it. The glasses run a Lens (`HandStream.ts`) that sends the
hands to the Mac over a WebSocket; the Mac (`trashdrop/spectacles.py`) has
the arms follow them. How a hand maps to its arm is in that module's
docstring.

## The Mac

    uv run python -m trashdrop.spectacles --dry-run   # first: no arms, just the link and where they would go
    uv run python -m trashdrop.spectacles             # then the arms (stop `trashdrop web` first: one program a bus)

It prints the address for the Lens (`ws://192.168.x.y:8765`). macOS may ask
whether Python may accept incoming connections: allow it. The glasses and
the Mac must be on the same Wi-Fi, and a network that keeps its devices
apart (many venue networks do) stops them seeing each other: a phone's
hotspot for both then.

Options: `--scale 1.5` (arm cm per hand cm), `--speed 120` (deg/s, the most
any joint turns), `--facing them` (standing in front of the arms, facing
them), `--arm left` (one arm only).

## The Lens (Lens Studio 5.9 or later)

1. Lens Studio signed in with the Snap account the glasses are paired with.
2. New project from a Spectacles template: it brings the Spectacles
   Interaction Kit (`SpectaclesInteractionKit.lspkg`).
3. Asset Browser: `+` > Internet Module.
4. Asset Browser: `+` > TypeScript File, named `HandStream`, with the contents
   of `HandStream.ts`.
5. Scene Hierarchy: a new scene object; Inspector: Add Component > Script >
   HandStream. Set Internet Module (the asset), Camera (the scene's Camera
   Object), Url (what the Mac printed). Status is optional: a Text (Screen
   Text or a world-space Text in front of the camera) to see what the arms do.
6. Project Settings: Experimental APIs on -- a plain `ws://` needs it. Such a
   Lens runs on your glasses but cannot be published; that is fine here.
7. Send it to the glasses with Preview Lens (USB-C, or both on the same
   Wi-Fi; Spectacles app: Developer Settings > Lens Development for USB).

In the Lens, look at your hands: the Status text says `connected`, then what
each arm is doing. Hands out of view hold the arms; Ctrl+C on the Mac holds
them too.
